import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist

from scipy.optimize import linear_sum_assignment
from itertools import permutations, combinations
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import glob
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../'))

from utils.var import Variable
from main.setup import params
from utils.guide import minimize_entropy_of_sinkhorn, sinkhorn, assert_shape, loft, permute_dims, compute_leakage_loss
from utils.guide import DisentanglementPrior, LabelEmbeddings, MLP, MixturePrior, MMDLoss, TCDiscriminator, ShallowMLP

import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(params["running_type"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def build_coords(b_s, resolution=(128, 128)):
    # x-cor and y-cor setting
    nx, ny = resolution
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    xv = torch.from_numpy(np.reshape(xv, [b_s, nx, nx, 1])).to(device, dtype=torch.float32)
    yv = torch.from_numpy(np.reshape(yv, [b_s, ny, ny, 1])).to(device, dtype=torch.float32)
    return xv, yv

class SlotAttention(nn.Module):
    def __init__(self, num_slots, slot_dim = 64, dim = 64, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slot_dim = slot_dim
        
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, slot_dim))

        self.to_q = nn.Linear(slot_dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, slot_dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(slot_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, slot_dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

        self.mlp_weight_input = nn.Linear(dim, 1)
        self.mlp_weight_slots = nn.Linear(slot_dim, 1)
        
        self.eps = torch.tensor(1e-8, device=device)

    def forward(self, inputs, num_slots = None):
        
        b_s, n, d = inputs.shape # 'inputs' have shape (b_s, W*H, dim)
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b_s, n_s, -1)
        sigma = self.slots_sigma.expand(b_s, n_s, -1)
        slots = torch.distributions.Normal(mu, torch.abs(sigma) + self.eps).rsample() # 'slots' have shape (1, n_s, slot_dim=64)
 
        inputs = self.norm_input(inputs)      
        k, v = self.to_k(inputs), self.to_v(inputs) # 'k' and 'v' have shape (1, 16384, 64)

        if params["strided_convs"]: latent_resolution = (32, 32)
        else: latent_resolution = (128, 128)
        
        x_pos, y_pos = build_coords(1, latent_resolution)
        grid = torch.cat([x_pos, y_pos], dim=-1)
        grid = torch.flatten(grid, -3, -2) # 'grid' has shape (1, 16384, 2)
        #grid_per_slot = grid.unsqueeze(-3).repeat(1, n_s, 1, 1) # 'grid_per_slot' has shape (1, n_s, 16384, 2)

        a = self.mlp_weight_input(inputs).squeeze(-1).softmax(-1) * n_s # 'a' shape (b_s, W*H)

        for iteration in range(self.iters):

            if params["running_type"] == "inspect": logging.info(f"\nSA iteration {iteration}")
            
            slots_prev = slots
            slots = self.norm_slots(slots) # 'slots' shape (b_s, n_s, slot_dim)
            b = self.mlp_weight_slots(slots).squeeze(-1).softmax(-1) * n_s  # 'b' shape (b_s, n_s)
            
            q = self.to_q(slots) # 'q' shape (1, n_s, 64)

            attn_logits = torch.cdist(k, q)      
            attn_logits, p, q = minimize_entropy_of_sinkhorn(attn_logits, a, b, mesh_lr=params["mesh_lr"], n_mesh_iters=params["mesh_iters"]) 
                    
            attn, _, _ = sinkhorn(attn_logits, a, b, u=p, v=q) # 'attn' shape (1, n, n_s)
            attn = attn.permute(0, 2, 1) # 'attn' shape (1, n_s, n)
            updates = torch.matmul(attn, v)

            # Compute the center of mass of each slot attention mask.
            # 'attn' has shape (1, n_s, 16384)
            # 'grid' has shape (1, 16384, 2)
            # 'slot_pos' has shape (1, n_s, 2)
            slot_pos = torch.einsum("...qk,...kd->...qd", attn, grid)
            slot_pos = slot_pos.unsqueeze(-2)  

            slots = self.gru(
                    updates.view(b_s * n_s, d),
                    slots_prev.view(b_s * n_s, self.slot_dim),
            )

            slots = slots.view(b_s, n_s, self.slot_dim)
            assert_shape(slots.size(), (b_s, n_s, self.slot_dim))
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
            assert_shape(slots.size(), (b_s, n_s, self.slot_dim))    
        return slots, slot_pos.reshape((b_s, n_s, 2)), attn#, scales

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0) # (1, 2, 128, 128)
    grid = grid.astype(np.float32)
    grid = torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)
    return grid

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()        
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)        

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        #logging.info(inputs.shape) # (1, 128, 128, 64)
        #logging.info(grid.shape)   # (1, 128, 128, 8)
        new_embd = inputs + grid
        return new_embd

class Encoder(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        
        self.encoder_sa = []
        in_channels = 3 if params['dataset'] == '2Dobjects' else 4
        self.encoder_sa += [nn.Conv2d(in_channels, 64, 5, 1, 2), nn.ReLU(inplace=True)]
        for c in range(2):
            if params["strided_convs"]: self.encoder_sa += [nn.Conv2d(64, 64, 5, 2, 2), nn.ReLU(inplace=True)]
            else: self.encoder_sa += [nn.Conv2d(64, 64, 5, 1, 2), nn.ReLU(inplace=True)]
        self.encoder_sa += [nn.Conv2d(64, 64, 5, 1, 2), nn.ReLU(inplace=True)]
        self.encoder_sa = nn.Sequential(*self.encoder_sa)
        
        #self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)
        if params["strided_convs"]: resolution = (32, 32)
        else: resolution = (128, 128)
        
        self.last_dim = 64
        self.encoder_pos = SoftPositionEmbed(self.last_dim, resolution)


    def forward(self, x):
        x = self.encoder_sa(x)
        x = x.permute(0,2,3,1) # B, W, H, C -> put C in last dimension
        x = self.encoder_pos(x)                 
        # spatial flatten -> flatten the spatial dimensions
        x = torch.flatten(x, 1, 2) # B, W*H, C -> 64 features of size w*H

        return x

"""Slot Attention-based auto-encoder for object discovery."""
class InvSlotAttentionGuide(nn.Module):
    def __init__(self, resolution, num_iterations, slot_dim, latent_dim, nlatents, A, stage):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.slot_dim = slot_dim
        self.resolution = resolution
        self.num_slots = 0
        self.num_iterations = num_iterations
        
        # self.shape_vals = params["shape_vals"]
        # self.color_vals = params["color_vals"]
        # self.size_vals = params["size_vals"]
        self.stage = stage
        assert self.stage in ["train", "eval"], "stage must be either 'train' or 'eval'"
        self.current_trace = []
        self.is_train = True
        self.pred_loss = 0.

        self.encoder_cnn = Encoder(self.resolution)

        self.hid_dim = self.encoder_cnn.last_dim
        self.ldim = latent_dim

        self.mlp = nn.Sequential(
             nn.Linear(self.hid_dim, self.hid_dim), # used to be hid_dim, hid_dim
             nn.ReLU(inplace=True),
             nn.Linear(self.hid_dim, self.hid_dim)
        )

        self.slot_attention = SlotAttention(
                num_slots=self.num_slots,
                slot_dim = self.slot_dim,
                dim=self.hid_dim,
                iters = self.num_iterations,
                eps = 1e-8, 
                hidden_dim = 128)

        self.prop_nets = nn.ModuleDict() 
        self.z_nets = nn.ModuleDict() 
        
        self.L = nlatents
        self.A = A
        self.num_classes_list = [len(params[f"{v}_vals"]) for v in params["latent_vars"]]
        self.num_classes_dict = {v: len(params[f"{v}_vals"]) for v in params["latent_vars"]}

        
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.gamma = 0.
        self.pred_coeff = 0.
        self.margin = 1.
        self.yc_t = 0.
        self.logvar_coeff = 0.
        self.SA_pretrain = False
        self.TC_coeff = params["TC_coeff"]
        self.log_det_coeff = params["log_det_coeff"]
        self.causal_dropout_prob = params["causal_dropout_prob"] if not params["causal_dropout_scheduled"] else 1.
        
        self.some_labels = torch.tensor([])
        self.gmm_fit_step = 200

        self.dag = self.A
        self.causal_flag = torch.zeros(params["batch_size"], params["num_slots"], 1).to(device)

        self.label_embeddings = LabelEmbeddings(num_classes_list=self.num_classes_list, latent_dim = self.ldim)
        
        if self.beta > 0.:
            self.prior = DisentanglementPrior(dim=self.L, k=self.ldim, C=self.A)
            self.mmd_loss = MMDLoss()

        self.s_cond = MLP(self.L*self.ldim, self.ldim, 100, f_act="tanh")
        self.t_cond = MLP(self.L*self.ldim, self.ldim, 100, f_act="tanh")

        self.batch_idx = 0
        self.step = 0
        
        if params["TC_minimization"]: self.tc_discriminator = TCDiscriminator(self.L*self.ldim, self.L*self.ldim*4)

        if params["reduce_noise_leak_adv"]:
            self.disentangle_mlps = nn.ModuleDict()
            for perm in permutations(params["latent_vars"], 2):
                self.disentangle_mlps[f"{perm[0]}_{perm[1]}"] = ShallowMLP(nin = self.ldim,
                                                                                         nout = self.num_classes_dict[perm[1]],
                                                                                         nh = self.ldim*2)
            
            self.rec_loss = nn.MSELoss()
    
    def add_proposal_net(self, var, out_dim):    
        add_flag = False
        if var.proposal_distribution == "categorical": last_activ = nn.Softmax(dim=-1) # discrete variables
        elif var.proposal_distribution == "bernoulli": last_activ = nn.Sigmoid() # mask
        else: raise ValueError(f"Unknown distribution: {var.proposal_distribution}")

        if params["pos_from_attn"] == "attn-masks": 
            input_dim = 1 if var.address in ["locX", "locY"] else self.slot_dim
        
        z_net = nn.Sequential(
            nn.Linear(input_dim, self.hid_dim), nn.ReLU(),
            nn.Linear(self.hid_dim, self.ldim)
            )
        
        if var.proposal_distribution != "categorical":
            self.z_nets[var.address] = z_net.to(device)
        else:
            #z_noise_net = NoiseNet(input_dim, self.hid_dim, self.ldim) # outputs mean and logvar of the representation
            z_noise_net = MixturePrior(label = [var.address],
                                                                 num_classes_list = [out_dim],
                                                                 nin = input_dim,
                                                                 nh = input_dim*4,
                                                                 ldim = self.ldim
                                                                 )
            self.z_nets[var.address] = z_noise_net.to(device)

        if params["conditioned_classifier"] and var.proposal_distribution != "bernoulli":
            proposal_net = nn.Sequential(
                nn.Linear(self.ldim + 1, out_dim), last_activ
                )
        else:
            proposal_net = nn.Sequential(
                nn.Linear(self.ldim, out_dim), last_activ
                )
        self.prop_nets[var.address] = proposal_net.to(device) 
        
        # if var.proposal_distribution == "categorical":
        #   noise_net = nn.Sequential(
        #     nn.Linear(self.hid_dim, self.hid_dim*2), nn.ReLU(),
        #     nn.Linear(self.hid_dim*2, self.hid_dim)
        #     )
        #   self.noise_nets[var.address] = noise_net.to(device) 

        if var.address in self.prop_nets: add_flag = True 
        if not add_flag: logging.info(f"ERROR: proposal net for site {var.name} was not added!")
    
    def infer_step(self, variable, obs=None, proposal_distribution=None): 

        # evaluation
        if isinstance(variable, str): 
            variable_name = variable      
            variable_proposal_distribution = proposal_distribution
        
        # training
        elif isinstance(variable, Variable): 
            variable_name = variable.name
            variable_address = variable.address
            variable_prior_distribution = variable.prior_distribution
            variable_proposal_distribution = variable.proposal_distribution
                
        if variable_proposal_distribution == "normal":
            z = None
            proposal = obs.squeeze(-1)
            # std = params['loc_proposal_std']
            # if self.stage == 'eval':
            #   proposal = proposal.expand(params['num_inference_samples'], -1)
            #   proposal = torch.clamp(proposal, 0.15, 0.85)
            #out = pyro.sample(variable_name, dist.Normal(proposal, std))

        elif variable_proposal_distribution == "categorical":  
            
            
            z = self.z_nets[variable_name](obs)
            proposal = None
            
            # if self.stage == 'eval': proposal = proposal.expand(params['num_inference_samples'], -1, -1)
            # out = pyro.sample(variable_name, dist.Categorical(probs=proposal))

        elif variable_proposal_distribution == "bernoulli": 
            z = None

            # if self.stage == "train":
            #   noise = torch.randn(obs.size()).to(device)
            #   obs = obs + noise
            
            proposal = self.prop_nets[variable_name](self.z_nets[variable_name](obs))
            proposal = proposal.squeeze(-1) 
    
            # if self.stage == 'eval': proposal = proposal.expand(params['num_inference_samples'], -1, -1)
            #out = pyro.sample(variable_name, dist.Bernoulli(proposal))
                        
        else: raise ValueError(f"Unknown variable proposal distribution: {variable_proposal_distribution}")

        return z, proposal
    
    def forward(self, observations={"image": torch.zeros((1, 3, 128, 128))}, N=None):

        # register networks to be optimized
        pyro.module("encoder", self.encoder_cnn, True)
        pyro.module("proposal_nets", self.prop_nets, True)
        pyro.module("z_nets", self.z_nets, True)
        pyro.module("sa", self.slot_attention, True)
        pyro.module("mlp", self.mlp, True)
        pyro.module("s_cond", self.s_cond, True)
        pyro.module("t_cond", self.t_cond, True)
        pyro.module("label_embeddings", self.label_embeddings, True)
        if params["beta"] > 0.: pyro.module("prior", self.prior, True)
        if params["TC_minimization"]: pyro.module("tc_discriminator", self.tc_discriminator, True)
        if params["reduce_noise_leak_adv"]: pyro.module("disentangle_mlps", self.disentangle_mlps, True)


        self.img = observations["image"]
        self.img = self.img.to(device)
        x = self.encoder_cnn(self.img)
        x = nn.LayerNorm(x.shape[1:]).to(device)(x)
        self.features_to_slots = self.mlp(x)

        B, _, _, _ = self.img.shape
        n_s = params['num_slots']  
          
        if self.stage == "train":
            self.slots, self.slot_pos, attn = self.slot_attention(self.features_to_slots, num_slots=n_s)
            if self.is_train and self.step % params['step_size_img_sv'] == 0:
                aux_attn = attn.reshape((B, n_s, 128, 128)) if not params["strided_convs"] else attn.reshape((B, n_s, 32, 32))
                fig, ax = plt.subplots(ncols=n_s)
                for j in range(n_s):                                       
                        im = ax[j].imshow(aux_attn[0, j, :, :].detach().cpu().numpy())
                        ax[j].grid(False)
                        ax[j].axis('off')        
                plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.step}/attn.png")
                plt.close()

                plot_img = np.transpose(self.img[0].detach().cpu().numpy(), (1, 2, 0))
                plt.imshow(np.clip(plot_img, 0., 1.))
                plt.axis('off')
                plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.step}/img.png")
                plt.close()
                        
            z_noise, mu_noise, logvar_noise, rec_noise = [], [], [], []
            proposals = {}
            labels_dict = {}
            for var in self.current_trace:
                if var.name == "mask":
                    real_mask = var.value

                if var.proposal_distribution == "categorical":
                    labels_dict[var.name] = var.value.long()

                # `slots` has shape: [batch_size, num_slots, slot_size].
                if var.address == "locX": obs = self.slot_pos[..., 0]
                elif var.address == "locY": obs = self.slot_pos[..., 1]
                else: 
                    obs = self.slots # [B, N, d]
                
                if not params["joint_training"]:
                    obs = obs.detach()
            
                # gather entangled posterior proposals and representations
                z_temp, proposal = self.infer_step(var, obs) 
                proposals[var.name] = proposal
                if isinstance(z_temp, tuple): 
                    mu, logvars, x_hat, z = z_temp
                    z_noise.append(z)
                    mu_noise.append(mu)
                    logvar_noise.append(logvars)
                    rec_noise.append(x_hat)
            z_noise = torch.stack(z_noise).permute(1, 2, 0, 3)            # [B, N, L, d] => 'L' refers to the number of categorical latents
            mu_noise = torch.stack(mu_noise).permute(1, 2, 0, 3)          # [B, N, L, d]
            logvar_noise = torch.stack(logvar_noise).permute(1, 2, 0, 3)  # [B, N, L, d]
            rec_noise = torch.stack(rec_noise).permute(1, 2, 0, 3)        # [B, N, L, slot_dim]
            assert z_noise.size(2) == self.L
    

            labels = torch.cat(([labels_dict[f"{v}"].unsqueeze(-1) for v in params["latent_vars"]]), dim = -1) # [B, N, L]
            u_m = self.label_embeddings(labels.reshape(B*n_s, self.L))
            l_to_label = {i: l for i, l in enumerate(params["latent_vars"])}


            self.kl_noise = 0.
            self.noise_loss = 0.
            self.label_embeddings_loss = 0.
            self.logvar_div = 0.
            self.mu_dist = 0.
            self.tc_loss = 0.
            self.noise_leak_encoder_loss = 0.
            self.log_det = 0.
            self.log_det_prior = 0.
            self.causal_loss = 0.
            self.logvar_noise = torch.zeros(B*n_s, self.L, self.ldim)


            # before fitting the GMM, only label embeddings is trained
            if self.step < self.gmm_fit_step:
                if self.is_train: 
                    for l in range(self.L):
                        dist_matrix = torch.cdist(u_m[:, l, :], u_m[:, l, :], p=2)/u_m.size(-1)
                        labels_l = labels[:, :, l].reshape(B*n_s, 1)
                        labels_matrix = (labels_l == labels_l.T).float()
                        pos_loss = labels_matrix * torch.square(dist_matrix)
                        neg_loss = (1-labels_matrix) * torch.square(F.relu(self.margin - dist_matrix))
                        self.label_embeddings_loss += torch.mean(pos_loss) + torch.mean(neg_loss)
                    
                    # collect some labels to then use for GMM fit
                    self.some_labels = torch.cat((self.some_labels, labels.reshape(B*n_s, self.L)), dim = 0)


            # now train the rest of the components
            else:
                
            
                # fit GMM 
                if self.is_train and self.step == self.gmm_fit_step:          
                    gmm_wts, gmm_means, gmm_stds = [], [], []
                    some_embeddings_m = self.label_embeddings(self.some_labels.long())
                    for l in range(self.L):
                        K = self.num_classes_list[l]
                        gmm = GaussianMixture(n_components=K, covariance_type='diag')
                        gmm.fit(some_embeddings_m[:, l, :].detach().cpu().numpy())
                        u_wts, u_means, u_vars = gmm.weights_, gmm.means_, gmm.covariances_    # [K], [K, ldim], [K, ldim]
                        preds = torch.from_numpy(gmm.predict(some_embeddings_m[:, l, :].detach().cpu().numpy())).to(device).to(torch.float) # [B]

                        conf_mat = torch.zeros((K, K), dtype=torch.int32)
                        for i in range(K):
                            for j in range(K):
                                conf_mat[i, j] = torch.sum((preds == i) & (self.some_labels[:, l] == j))

                        # Apply Hungarian algorithm to maximize agreement
                        row_ind, col_ind = linear_sum_assignment(-conf_mat.cpu().numpy())  # maximize by negating
                        mapping = {row: col for row, col in zip(row_ind, col_ind)}            
                        
                        # pass to tensor
                        u_wts = torch.from_numpy(u_wts).to(device)
                        u_means = torch.from_numpy(u_means).to(device) 
                        u_vars = torch.from_numpy(u_vars).to(device)

                        u_wts = torch.index_select(u_wts, dim=0, index = torch.tensor(list(mapping.values())))
                        u_means = torch.index_select(u_means, dim=0, index = torch.tensor(list(mapping.values())))
                        u_vars = torch.index_select(u_vars, dim=0, index = torch.tensor(list(mapping.values())))

                        gmm_wts.append(u_wts)
                        gmm_means.append(u_means)
                        gmm_stds.append(torch.sqrt(u_vars))

                        # if self.is_train and self.step % params['step_size'] == 0:
                        #   logger.info(f"\nGMM mixture params for variable {l_to_label[l]}:")
                        #   logger.info(f"GMM wts: {u_wts}")
                        #   logger.info(f"GMM means: {u_means}")
                        #   logger.info(f"GMM vars: {u_vars}")
                    
                        # set the prior params in MixturePrior according to what is inferred by the GMM
                        self.z_nets[l_to_label[l]].pi_c = u_wts
                        self.z_nets[l_to_label[l]].mu_c = u_means
                        self.z_nets[l_to_label[l]].logvar_c = torch.log(u_vars)

                else:
                    gmm_wts, gmm_means, gmm_stds = [], [], []
                    for l in range(self.L):
                        gmm_wts.append(self.z_nets[l_to_label[l]].pi_c)
                        gmm_means.append(self.z_nets[l_to_label[l]].mu_c)
                        gmm_stds.append(torch.exp(0.5 * self.z_nets[l_to_label[l]].logvar_c))

                u_m = u_m.reshape(B, n_s, self.L, self.ldim)

                if params["causal_dropout_prob"] < 1.:
                    z_scm = torch.zeros_like(z_noise).to(device)
                    total_dims = self.L*self.ldim
                    L_slopes = torch.tensor([]).to(device)
                    for i in range(self.L): 
                        if 1 in self.dag[i, :]:
                            mask = self.dag[i, :].repeat(self.ldim, 1).T.reshape(total_dims).to(device)
                            # s = self.s_cond(z_scm.reshape(-1, total_dims), mask).reshape(B, n_s, self.ldim) # slope
                            # t = self.t_cond(z_scm.reshape(-1, total_dims), mask).reshape(B, n_s, self.ldim) # offset
                        
                        else:
                            mask = torch.zeros(total_dims).to(device)
                            # s = torch.zeros(B, n_s, self.ldim).to(device) # slope
                            # t = torch.zeros(B, n_s, self.ldim).to(device) # offset
                        
                        s = self.s_cond(z_scm.reshape(-1, total_dims), mask).reshape(B, n_s, self.ldim) # slope
                        t = self.t_cond(z_scm.reshape(-1, total_dims), mask).reshape(B, n_s, self.ldim) # offset

                        # constrain slopes
                        if params["slope_constraint_flag"]: s = loft(s)
                        
                        z_i =  torch.exp(s) * z_noise[:, :, i, :] + t
                        z_scm[:, :, i, :] = z_i

                        s = s.reshape(B*n_s, self.ldim)
                        t = t.reshape(B*n_s, self.ldim)
                        
                        L_slopes = torch.cat((L_slopes, s.unsqueeze(-2)), dim=-2)
                else:
                    z_scm = z_noise
                
                discrete_latent_proposals = {}
                var_idx = {l: i for i, l in enumerate(params["latent_vars"])}
                # now need to compute proposals and make predictions with the causal disentangled representations 
                for var in self.current_trace:
                    if var.proposal_distribution == "normal": 
                        proposal = proposals[var.name]
                        pyro.sample(var.name, dist.Normal(proposal, params['loc_proposal_std']))
                    elif var.proposal_distribution == "bernoulli": 
                        proposal = proposals[var.name]
                        mask_pred = proposal.reshape(B*n_s).detach()
                        pyro.sample(var.name, dist.Bernoulli(proposal))
                    elif var.proposal_distribution == "categorical": 
                        idx = var_idx[var.name] 
                        
                        if params["conditioned_classifier"]:
                            pred_input = torch.cat((z_scm[:, :, idx, :], self.causal_flag), dim=-1)
                        else:
                            pred_input = z_scm[:, :, idx, :]
                        proposal = self.prop_nets[var.name](pred_input)
                        discrete_latent_proposals[var.name] = proposal

                        assert not torch.isnan(z_noise[:, :, idx]).any(), f"{idx}"
                        assert not torch.isnan(z_scm[:, :, idx]).any(), f"{idx}"
                        assert not torch.isnan(proposal).any()

                        pyro.sample(var.name, dist.Categorical(probs=proposal))
                        
                    # log predictions
                    if self.step % params['step_size'] == 0:
                        logger.info(f"\n{var.name} target values {var.value[0]}")
                        logger.info(f"\n{var.name} proposed values {proposal[0]}")
                
                if self.is_train:
                    z_noise = z_noise.reshape(B*n_s, self.L, self.ldim)
                    mu_noise = mu_noise.reshape(B*n_s, self.L, self.ldim)
                    logvar_noise = logvar_noise.reshape(B*n_s, self.L, self.ldim)
                    rec_noise = rec_noise.reshape(B*n_s, self.L, self.slot_dim)

                    self.logvar_noise = logvar_noise
                    
                    preds = torch.tensor([]) # [B, N, L, 3]
                    for l in range(self.L):
                        preds = torch.cat(
                            (preds, discrete_latent_proposals[l_to_label[l]].unsqueeze(-2)),
                            dim = -2)

                    print_mixt_params = False
                    for l in range(self.L):
                        if self.is_train and self.step % params["step_size"] == 0: 
                            if self.step >= self.gmm_fit_step:
                                logger.info(f"\nGMMVAE losses for variable {l_to_label[l]}:")
                                print_mixt_params = True
                        
                        rec_loss, mixt_kl, kl_reg, yc = self.z_nets[l_to_label[l]].compute_elbo(mu_noise[:, l, :].unsqueeze(-2),
                                                                                                                                                                logvar_noise[:, l, :].unsqueeze(-2),
                                                                                                                                                                z_noise[:, l, :],
                                                                                                                                                                rec_noise[:, l, :],
                                                                                                                                                                self.slots.reshape(B*n_s, self.slot_dim),
                                                                                                                                                                mask_pred,
                                                                                                                                                                print_mixt_params=print_mixt_params,
                                                                                                                                                                yc_temperature=self.yc_t)
                        
                        if self.is_train and self.step % params["step_size"] == 0 and self.step >= self.gmm_fit_step:
                            #logger.info(f"yc shape: {yc.shape}")
                            
                            logger.info(f"yc: {torch.mean(yc, dim=0).tolist()}")
                            logger.info(f"pi_c: {self.z_nets[l_to_label[l]].pi_c.tolist()}")
                            logger.info(f"logvars: {torch.mean(logvar_noise[:, l, :])}")
                            logger.info(f"logvar_c: {torch.mean(self.z_nets[l_to_label[l]].logvar_c, dim=-1)}")          
                        
                        self.kl_noise += rec_loss + 0.01 * mixt_kl - kl_reg

                        K = discrete_latent_proposals[l_to_label[l]].shape[-1] 
                        real_idx = torch.where(torch.round(mask_pred) == 1.)[0]
                        for k in range(K):
                            k_idx = torch.where((torch.argmax(discrete_latent_proposals[l_to_label[l]].reshape(B*n_s, -1), dim=-1) == k))[0]
                            intersect = [a for a in k_idx if a in real_idx]
                            
                            if len(intersect) > 0:
                                k_idx_intersect = torch.stack(intersect)
                                self.mu_dist += torch.mean(torch.cdist(mu_noise[k_idx_intersect, l, :], mu_noise[k_idx_intersect, l, :]))

                        # add a constraint to approximate encoded to prior logvars, using the KL-div between to zero-mean Gaussians
                        
                        if params["noise_logvar_constraint"]:
                            l_logvar_div = torch.mean(self.z_nets[l_to_label[l]].kl_div_mixt(
                                logvar_noise[:, l, :],
                                self.z_nets[l_to_label[l]].logvar_c
                                ))
                            self.logvar_div += l_logvar_div

                    if params["TC_minimization"]:
                        for name, param in self.tc_discriminator.named_parameters():
                            assert param.requires_grad == False
                        
                        #disc_input = mask_pred.unsqueeze(-1)*torch.flatten(z_noise, -2)
                        disc_input = torch.flatten(z_noise, -2)
                        self.disc_input = disc_input
                        self.D_joint = self.tc_discriminator(disc_input)
                        self.tc_loss = torch.square(torch.mean(self.D_joint[:, :1] - self.D_joint[:, 1:]))
                        self.fake_z_noise, perms = permute_dims(disc_input, self.ldim)
                    

                    if params["reduce_noise_leak_adv"]:
                        for perm in permutations(params["latent_vars"], 2):
                            for name, param in self.disentangle_mlps[f"{perm[0]}_{perm[1]}"].named_parameters():
                                assert param.requires_grad == False

                        for lp in permutations(params["latent_vars"], 2):
                            lp_input = z_noise.reshape(B, n_s, self.L, self.ldim)[:, :, var_idx[lp[0]], :]  # [B, N, ldim]
                            lp_target = labels[:, :, var_idx[lp[1]]]                                        # [B, N]  
                            leak_logits = self.disentangle_mlps[f"{lp[0]}_{lp[1]}"](lp_input)                            # [B, N, K]            
                            leak_loss = compute_leakage_loss(leak_logits, lp_target)
                            self.noise_leak_encoder_loss += 1/leak_loss

                        self.mask_pred = mask_pred
                        self.z_noise = z_noise
                        self.labels = labels

                    
                    if params["causal_dropout_prob"] < 1.:
                        for i in range(self.L):
                            s = L_slopes[:, i, :]
                            #self.log_det += torch.sum(mask_pred.unsqueeze(-1) * s, dim=-1)/s.size(-1)
                            self.log_det += torch.sum(s, dim=-1)/s.size(-1)
                        self.log_det = torch.mean(self.log_det)
                    
                    if params["reduce_noise_leak_adv"]:
                        self.noise_loss = self.kl_noise + 0.5*(self.tc_loss + self.noise_leak_encoder_loss) - self.log_det
                    else:
                        self.noise_loss = self.kl_noise + self.TC_coeff*self.tc_loss - self.log_det_coeff*self.log_det

                    if self.beta > 0:
                        cp_m, log_det_prior, aligned_u_m, self.attn_loss, attn = self.prior(u_m, z_scm, self.dag, preds, labels) # [B, N, L, d]
                        self.log_det_prior = torch.mean(log_det_prior)
                    
                        for i in range(self.L):
                            #self.causal_loss += torch.mean(kl_normal(z_scm[:, :, i, :], cp_v, cp_m[:, :, i, :], cp_v))
                            self.causal_loss += self.mmd_loss(z_scm[:, :, i, :].reshape(B*n_s, -1), cp_m[:, :, i, :].reshape(B*n_s, -1))     
                    
                #if self.is_train and self.step > 0 and (self.step % params['step_size_img_sv'] == 0 or self.step == self.gmm_fit_step):
                if False:
                    for l in range(self.L): 

                            # get indexes of objects predicted as 'real'
                            real_idx = torch.where(torch.round(mask_pred) == 1.)[0]
                            
                            if len(real_idx) > 0:
                            
                                preds = torch.argmax(discrete_latent_proposals[l_to_label[l]].reshape(B*n_s, -1), dim=-1)[real_idx].detach().cpu().numpy()
                            
                                """ permute labels according to the attention matrix obtained in self.prior(.) """
                                
                                # turn 'attn' into a hard permutation matrix
                                if self.beta > 0.:
                                    perm = torch.round(attn)
                                    #embed_labels = labels[:, :, l].reshape(B*n_s).detach().cpu().numpy() # [B*N]
                                    perm_labels = torch.matmul(perm, labels[:, :, l].float().unsqueeze(-1)).squeeze(-1).reshape(B*n_s).detach().cpu().numpy() # [B*N]
                                
                                # NOISE REPRESENTATIONS
                                
                                z_l = z_noise[real_idx, l, :].detach().cpu().numpy() # [B*N, d]
                                mu_l = mu_noise[real_idx, l, :].detach().cpu().numpy() # [B*N, d]

                                if params["TC_minimization"]:
                                    perm_real_idx = torch.where(torch.round(mask_pred[perms[l]]) == 1.)[0]
                                    fake_z_l = self.fake_z_noise.reshape(B*n_s, self.L, self.ldim)[perm_real_idx, l, :].detach().cpu().numpy()
                                    fake_preds = torch.argmax(discrete_latent_proposals[l_to_label[l]].reshape(B*n_s, -1), dim=-1)[perms[l]][perm_real_idx].detach().cpu().numpy()

                                #logger.info(f"fake preds: {fake_preds[:5]}")


                                # Apply PCA to reduce dimensionality to 2D
                                if self.ldim != 2:
                                    #pca = PCA(n_components=2)
                                    pca = TSNE(n_components=2, random_state=0)
                                    z_pca = pca.fit_transform(z_l)  # Output shape: (sum(num_classes_per_label), 2)
                                    mu_pca = pca.fit_transform(mu_l)
                                    if params["TC_minimization"]: 
                                        fake_z_pca = pca.fit_transform(fake_z_l)
                                else:
                                    z_pca = z_l
                                    mu_pca = mu_l

                                # Plot with class indices as color
                                plt.figure(figsize=(8, 6))
                                scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=preds, alpha=0.5)
                                # Add legend
                                plt.legend(handles=scatter.legend_elements()[0], labels=[f"class {c}" for c in range(self.num_classes_list[l])], title="Class")
                                #plt.colorbar(label="Class Index")
                                plt.title("exogenous noise representations (z)")
                                plt.xlabel("PC1")
                                plt.ylabel("PC2")
                                plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.step}/z_noise_{l_to_label[l]}.png")
                                plt.close()

                                if l == 1:
                                    
                                    # get color preds
                                    color_preds = torch.argmax(discrete_latent_proposals[l_to_label[0]].reshape(B*n_s, -1), dim=-1)[real_idx].detach().cpu().numpy()
                                    
                                    plt.figure(figsize=(8, 6))
                                    scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=color_preds, alpha=0.5)
                                    # Add legend
                                    plt.legend(handles=scatter.legend_elements()[0], labels=[f"class {c}" for c in range(self.num_classes_list[l])], title="Class")
                                    #plt.colorbar(label="Class Index")
                                    plt.title("exogenous noise representations (z)")
                                    plt.xlabel("PC1")
                                    plt.ylabel("PC2")
                                    plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.step}/z_noise_{l_to_label[l]}_with_color_preds.png")
                                    plt.close()
                                
                                if l == 0:
                                    
                                    # get color preds
                                    shape_preds = torch.argmax(discrete_latent_proposals[l_to_label[1]].reshape(B*n_s, -1), dim=-1)[real_idx].detach().cpu().numpy()
                                    
                                    plt.figure(figsize=(8, 6))
                                    scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=shape_preds, alpha=0.5)
                                    # Add legend
                                    plt.legend(handles=scatter.legend_elements()[0], labels=[f"class {c}" for c in range(self.num_classes_list[l])], title="Class")
                                    #plt.colorbar(label="Class Index")
                                    plt.title("exogenous noise representations (z)")
                                    plt.xlabel("PC1")
                                    plt.ylabel("PC2")
                                    plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.step}/z_noise_{l_to_label[l]}_with_shape_preds.png")
                                    plt.close()


                                # Plot with class indices as color
                                if params["TC_minimization"]:
                                    plt.figure(figsize=(8, 6))
                                    scatter = plt.scatter(fake_z_pca[:, 0], fake_z_pca[:, 1], c=fake_preds, alpha=0.5)
                                    # Add legend
                                    plt.legend(handles=scatter.legend_elements()[0], labels=[f"class {c}" for c in range(self.num_classes_list[l])], title="Class")
                                    #plt.colorbar(label="Class Index")
                                    plt.title("fake exogenous noise representations (z)")
                                    plt.xlabel("PC1")
                                    plt.ylabel("PC2")
                                    plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.step}/fake_z_noise_{l_to_label[l]}.png")
                                    plt.close()

                                plt.figure(figsize=(8, 6))
                                scatter = plt.scatter(mu_pca[:, 0], mu_pca[:, 1], c=preds, alpha=0.5)
                                # Add legend
                                plt.legend(handles=scatter.legend_elements()[0], labels=[f"class {c}" for c in range(self.num_classes_list[l])], title="Class")
                                #plt.colorbar(label="Class Index")
                                plt.title("exogenous noise representations (mu)")
                                plt.xlabel("PC1")
                                plt.ylabel("PC2")
                                plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.step}/mu_noise_{l_to_label[l]}.png")
                                plt.close()  

                                if l == 0:
                                    
                                    # get color preds
                                    shape_preds = torch.argmax(discrete_latent_proposals[l_to_label[1]].reshape(B*n_s, -1), dim=-1)[real_idx].detach().cpu().numpy()
                                    
                                    plt.figure(figsize=(8, 6))
                                    scatter = plt.scatter(mu_pca[:, 0], mu_pca[:, 1], c=shape_preds, alpha=0.5)
                                    # Add legend
                                    plt.legend(handles=scatter.legend_elements()[0], labels=[f"class {c}" for c in range(self.num_classes_list[l])], title="Class")
                                    #plt.colorbar(label="Class Index")
                                    plt.title("exogenous noise representations (z)")
                                    plt.xlabel("PC1")
                                    plt.ylabel("PC2")
                                    plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.step}/mu_noise_{l_to_label[l]}_with_shape_preds.png")
                                    plt.close()    

                                if l == 1:
                                    
                                    # get color preds
                                    color_preds = torch.argmax(discrete_latent_proposals[l_to_label[0]].reshape(B*n_s, -1), dim=-1)[real_idx].detach().cpu().numpy()
                                    
                                    plt.figure(figsize=(8, 6))
                                    scatter = plt.scatter(mu_pca[:, 0], mu_pca[:, 1], c=color_preds, alpha=0.5)
                                    # Add legend
                                    plt.legend(handles=scatter.legend_elements()[0], labels=[f"class {c}" for c in range(self.num_classes_list[l])], title="Class")
                                    #plt.colorbar(label="Class Index")
                                    plt.title("exogenous noise representations (z)")
                                    plt.xlabel("PC1")
                                    plt.ylabel("PC2")
                                    plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.step}/mu_noise_{l_to_label[l]}_with_color_preds.png")
                                    plt.close()     
                                
                                # LABEL EMBEDDINGS
                                z_l = u_m.reshape(B*n_s, self.L, self.ldim)[:, l, :].detach().cpu().numpy() # [B*N, d]
                                embed_labels = labels[:, :, l].reshape(B*n_s).detach().cpu().numpy() # [B*N]

                                # Apply PCA to reduce dimensionality to 2D
                                if self.ldim != 2:
                                    pca = PCA(n_components=2)
                                    z_pca = pca.fit_transform(z_l)  # Output shape: (sum(num_classes_per_label), 2)
                                    
                                    if self.step >= self.gmm_fit_step:
                                        gmm_means_pca = pca.fit_transform(gmm_means[l].detach().cpu().numpy())
                                        #gmm_stds_pca = pca.fit_transform(gmm_stds[l].detach().cpu().numpy())
                                else:
                                    z_pca = z_l
                                    if self.step >= self.gmm_fit_step:
                                        gmm_means_pca = gmm_means[l].detach().cpu().numpy()
                                        #gmm_stds_pca = gmm_stds[l].detach().cpu().numpy()

                                # Plot with class indices as color
                                fig, ax = plt.subplots(figsize=(8, 6))
                                scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=embed_labels, cmap='rainbow', alpha=0.3)
                                if self.step > self.gmm_fit_step:
                                    for j in range(gmm_means_pca.shape[0]):
                                        #color = "blue" if c == 0 else "red"
                                        ax.scatter(gmm_means_pca[j, 0], gmm_means_pca[j, 1], marker="x")
                                        # ellipse = Ellipse(gmm_means_pca[c], width=2*gmm_stds_pca[c, 0], height=2*gmm_stds_pca[c, 1], 
                                        #                               edgecolor=color, facecolor='none', linestyle='--', linewidth=2)
                                        # ax.add_patch(ellipse)

                                # Add legend
                                plt.legend(handles=scatter.legend_elements()[0], labels=[f"class {c}" for c in range(self.num_classes_list[l])], title="Class")
                                #plt.colorbar(label="Class Index")
                                plt.title("label embeddings u")
                                plt.xlabel("PC1")
                                plt.ylabel("PC2")
                                plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.step}/embeddings_{l_to_label[l]}.png")
                                plt.close()


                                if self.beta > 0:
                                    # ALIGNED LABEL EMBEDDINGS
                                    z_l = aligned_u_m.reshape(B*n_s, self.L, self.ldim)[:, l, :].detach().cpu().numpy() # [B*N, d]

                                    # Apply PCA to reduce dimensionality to 2D
                                    if self.ldim != 2:
                                        pca = PCA(n_components=2)
                                        z_pca = pca.fit_transform(z_l)  # Output shape: (sum(num_classes_per_label), 2)
                                        
                                        if self.step >= self.gmm_fit_step:
                                            gmm_means_pca = pca.fit_transform(gmm_means[l].detach().cpu().numpy())
                                            #gmm_stds_pca = pca.fit_transform(gmm_stds[l].detach().cpu().numpy())
                                    else:
                                        z_pca = z_l
                                        if self.step >= self.gmm_fit_step:
                                            gmm_means_pca = gmm_means[l].detach().cpu().numpy()
                                            #gmm_stds_pca = gmm_stds[l].detach().cpu().numpy()

                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=perm_labels, cmap='rainbow', alpha=0.3)
                                    if self.step > self.gmm_fit_step:
                                        for c in range(gmm_means_pca.shape[0]):
                                            #color = "blue" if c == 0 else "red"
                                            ax.scatter(gmm_means_pca[c, 0], gmm_means_pca[c, 1], marker="x", alpha=0.9)
                                            # ellipse = Ellipse(gmm_means_pca[c], width=2*gmm_stds_pca[c, 0], height=2*gmm_stds_pca[c, 1], 
                                            #                               edgecolor=color, facecolor='none', linestyle='--', linewidth=2)
                                            # ax.add_patch(ellipse)

                                    # Add legend
                                    plt.legend(handles=scatter.legend_elements()[0], labels=[f"class {c}" for c in range(self.num_classes_list[l])], title="Class")
                                    #plt.colorbar(label="Class Index")
                                    plt.title("label embeddings u")
                                    plt.xlabel("PC1")
                                    plt.ylabel("PC2")
                                    plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.step}/aligned_embeddings_{l_to_label[l]}.png")
                                    plt.close()
                                

                                # POSTERIOR
                                
                                z_l = z_scm.reshape(B*n_s, self.L, self.ldim)[real_idx, l, :].detach().cpu().numpy() # [B*N, d]

                                # Apply PCA to reduce dimensionality to 2D
                                if self.ldim != 2:
                                    #pca = PCA(n_components=2)
                                    pca = TSNE(n_components=2, random_state=0)
                                    z_pca = pca.fit_transform(z_l)  # Output shape: (sum(num_classes_per_label), 2)
                                else:
                                    z_pca = z_l

                                # Plot with class indices as color
                                plt.figure(figsize=(8, 6))
                                scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=preds, cmap='rainbow', alpha=0.5)
                                # Add legend
                                plt.legend(handles=scatter.legend_elements()[0], labels=[f"class {c}" for c in range(self.num_classes_list[l])], title="Class")
                                #plt.colorbar(label="Class Index")
                                plt.title("z_scm")
                                plt.xlabel("PC1")
                                plt.ylabel("PC2")
                                plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.step}/posterior_{l_to_label[l]}.png")
                                plt.close()


                                # CAUSAL PRIOR
                                if self.beta > 0:
                                    z_l = cp_m.reshape(B*n_s, self.L, self.ldim)[:, l, :].detach().cpu().numpy() # [B*N, d]

                                    # Apply PCA to reduce dimensionality to 2D
                                    if self.ldim != 2:
                                        #pca = PCA(n_components=2)
                                        pca = TSNE(n_components=2, random_state=0)
                                        z_pca = pca.fit_transform(z_l)  # Output shape: (sum(num_classes_per_label), 2)
                                    else:
                                        z_pca = z_l

                                    # Plot with class indices as color
                                    plt.figure(figsize=(8, 6))
                                    scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=perm_labels, cmap='rainbow', alpha=0.5)
                                    # Add legend
                                    plt.legend(handles=scatter.legend_elements()[0], labels=[f"class {c}" for c in range(self.num_classes_list[l])], title="Class")
                                    #plt.colorbar(label="Class Index")
                                    plt.title("PCA Projection of Learned Prior p(z|u)")
                                    plt.xlabel("PC1")
                                    plt.ylabel("PC2")
                                    plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.step}/prior_w_labels_{l_to_label[l]}.png")
                                    plt.close()

                                    
                                    # CAUSAL PRIOR 
                                    
                                    z_l = cp_m.reshape(B*n_s, self.L, self.ldim)[real_idx, l, :].detach().cpu().numpy() # [B*N, d]

                                    # Apply PCA to reduce dimensionality to 2D
                                    if self.ldim != 2:
                                        #pca = PCA(n_components=2)
                                        pca = TSNE(n_components=2, random_state=0)
                                        z_pca = pca.fit_transform(z_l)  # Output shape: (sum(num_classes_per_label), 2)
                                    else:
                                        z_pca = z_l

                                    # Plot with class indices as color
                                    plt.figure(figsize=(8, 6))
                                    scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=preds, cmap='rainbow', alpha=0.5)
                                    # Add legend
                                    plt.legend(handles=scatter.legend_elements()[0], labels=[f"class {c}" for c in range(self.num_classes_list[l])], title="Class")
                                    #plt.colorbar(label="Class Index")
                                    plt.title("PCA Projection of Learned Prior p(z|u)")
                                    plt.xlabel("PC1")
                                    plt.ylabel("PC2")
                                    plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.step}/prior_w_preds_{l_to_label[l]}.png")
                                    plt.close()

                                    if l == 1:
                                        
                                        # get color preds
                                        color_preds = torch.argmax(discrete_latent_proposals[l_to_label[0]].reshape(B*n_s, -1), dim=-1)[real_idx].detach().cpu().numpy()
                                        
                                        plt.figure(figsize=(8, 6))
                                        scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=color_preds, alpha=0.5)
                                        # Add legend
                                        plt.legend(handles=scatter.legend_elements()[0], labels=[f"class {c}" for c in range(self.num_classes_list[l])], title="Class")
                                        #plt.colorbar(label="Class Index")
                                        plt.title("exogenous noise representations (z)")
                                        plt.xlabel("PC1")
                                        plt.ylabel("PC2")
                                        plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.step}/prior_{l_to_label[l]}_with_color_preds.png")
                                        plt.close()
                                    
                                    if l == 0:
                                        
                                        # get color preds
                                        shape_preds = torch.argmax(discrete_latent_proposals[l_to_label[1]].reshape(B*n_s, -1), dim=-1)[real_idx].detach().cpu().numpy()
                                        
                                        plt.figure(figsize=(8, 6))
                                        scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=shape_preds, alpha=0.5)
                                        # Add legend
                                        plt.legend(handles=scatter.legend_elements()[0], labels=[f"class {c}" for c in range(self.num_classes_list[l])], title="Class")
                                        #plt.colorbar(label="Class Index")
                                        plt.title("exogenous noise representations (z)")
                                        plt.xlabel("PC1")
                                        plt.ylabel("PC2")
                                        plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.step}/prior_{l_to_label[l]}_with_shape_preds.png")
                                        plt.close()
                            

            del self.slots
            self.current_trace = []
        
        elif self.stage == "eval":
            with torch.no_grad():
                assert self.current_trace == [], "current_trace list is not empty in the begining of evaluation!"

                n_s = params['max_objects']
                self.slots, self.slot_pos, attn = self.slot_attention(self.features_to_slots, num_slots=n_s) 

                # the idea here is to reset the causal graph adjacency matrix
                # and use the final representations to compute predictions
                # - during training, the guide learns to extract representations that both
                #   align with causal representations and predict posterior params well
                # - for inference, in a compositional generalization setting, I just need to 
                #   extract representations that don't depend on any parent...

                if "size" in params["latent_vars"]: latents = ["mask", "color", "shape", "size", "locX", "locY"]
                else: latents = ["mask", "color", "shape", "locX", "locY"]
                
                z_noise = []
                for var in latents:
                    if var in ["mask"]: 
                        proposal_distribution = "bernoulli"
                        prior_distribution = "bernoulli"
                        obs = self.slots
                    elif var in ["shape", "color", "size"]: 
                        proposal_distribution = "categorical"
                        prior_distribution = "categorical"
                        obs = self.slots
                    elif var in ["locX", "locY"]: 
                        proposal_distribution = "normal"
                        prior_distribution = "uniform"
                        if var == "locX": obs = self.slot_pos[..., 0]
                        elif var == "locY": obs = self.slot_pos[..., 1]
                    
                    z_temp, proposal = self.infer_step(var, obs, proposal_distribution) 
                    
                    if proposal_distribution == "normal":
                        proposal = proposal.expand(params['num_inference_samples'], -1)
                        proposal = torch.clamp(proposal, 0.15, 0.85)
                        pyro.sample(var, dist.Normal(proposal, params['loc_proposal_std']))
                    
                    elif proposal_distribution == "bernoulli":
                        logger.info(f"PREDICTED MASK: {proposal}")
                        proposal = proposal.expand(params['num_inference_samples'], -1)
                        pyro.sample(var, dist.Bernoulli(proposal))

                    elif proposal_distribution == "categorical":
                        mu, logvars, x_hat, z = z_temp
                        z_noise.append(z)

                        #logger.info(f"\nappending noise representations of {var}...")
                
                z_noise = torch.stack(z_noise).permute(1, 2, 0, 3) # [B, N, L, d] => 'L' refers to the number of categorical latents
                assert z_noise.size(2) == self.L  
                B, N, L, d = z_noise.shape

                z_scm = torch.zeros_like(z_noise)
                total_dims = self.L*self.ldim
                for i in range(self.L): 
                    if 1 in self.A[i, :]:
                        mask = self.A[i, :].repeat(self.ldim, 1).T.reshape(total_dims).to(device)
                        
                    else:
                        mask = torch.zeros(total_dims).to(device)
                        # s = torch.zeros(B, n_s, self.ldim).to(device) # slope
                        # t = torch.zeros(B, n_s, self.ldim).to(device) # offset

                    s = self.s_cond(z_scm.reshape(-1, total_dims), mask).reshape(B, n_s, self.ldim) # slope
                    t = self.t_cond(z_scm.reshape(-1, total_dims), mask).reshape(B, n_s, self.ldim) # offset
                    
                    # constrain slopes
                    if params["slope_constraint_flag"]: s = loft(s)

                    assert not torch.isnan(s).any()
                    
                    z_i =  torch.exp(s) * z_noise[:, :, i, :] + t
                    z_scm[:, :, i, :] = z_i
                
                #var_idx = {"color": 0, "shape": 1, "size": 2}
                var_idx = {l: i for i, l in enumerate(params["latent_vars"])}
                latents = list(var_idx.keys())
                for var in latents:
                    idx = var_idx[var] 
                    
                    if params["conditioned_classifier"]:
                        
                        if torch.equal(self.A, torch.zeros(self.L, self.L)):
                            causal_flag = torch.ones(B, n_s, 1).to(device)
                        else:
                            causal_flag = torch.zeros(B, n_s, 1).to(device)
                        
                        pred_input = torch.cat((z_scm[:, :, idx, :], causal_flag), dim=-1)
                    else:
                        pred_input = z_scm[:, :, idx, :]
                    
                    proposal = self.prop_nets[var](pred_input)
                    
                    logger.info(f"\n{var} proposal given the intervention on the causal graph:")
                    logger.info(proposal)
                    
                    proposal = proposal.expand(params['num_inference_samples'], -1, -1)
                    pyro.sample(var, dist.Categorical(probs=proposal))

                    # new_var = Variable(name=var,
                    #                    value=out,
                    #                    prior_distribution=prior_distribution,
                    #                    proposal_distribution=proposal_distribution,
                    #                    address=var
                    #                    )
                    # self.current_trace.append(new_var)

                self.current_trace = []
                return
            
        else: raise ValueError(f"Unknown stage: {self.stage}")

