import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import glob
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

# add project path to sys to import relative modules
import sys
sys.path.insert(0, '../..') # add parent-parent dir

from src.main import params
from src.utils.guide import minimize_entropy_of_sinkhorn, sinkhorn, assert_shape
from src.utils.guide import VAE, MMDLoss, TCDiscriminator, MultivariateCausalFlow, PriorMultivariateCausalFlow

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

        latent_resolution = (32, 32)
        
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
            slots = slots + self.fc2(F.elu(self.fc1(self.norm_pre_ff(slots))))
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
        in_channels = 3 
        self.encoder_sa += [nn.Conv2d(in_channels, 64, 5, 1, 2), nn.ELU()]
        for c in range(2):
            self.encoder_sa += [nn.Conv2d(64, 64, 5, 2, 2), nn.ELU()]
        self.encoder_sa += [nn.Conv2d(64, 64, 5, 1, 2), nn.ELU()]
        self.last_dim = self.encoder_sa[-2].out_channels
        self.encoder_sa = nn.Sequential(*self.encoder_sa)
        
        #self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)
        resolution = (32, 32)        
        self.encoder_pos = SoftPositionEmbed(self.last_dim, resolution)


    def forward(self, x):
        x = self.encoder_sa(x)
        x = x.permute(0,2,3,1) # B, W, H, C -> put C in last dimension
        x = self.encoder_pos(x)                 
        # spatial flatten -> flatten the spatial dimensions
        x = torch.flatten(x, 1, 2) # B, W*H, C -> 64 features of size w*H
        return x

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    
class Decoder(nn.Module):
    def __init__(self, in_dim, out_channels):
        super().__init__()

        module_list = [
            nn.Linear(in_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 32 * 4 * 4),
            nn.ELU(),
            View((-1, 32, 4, 4)),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ELU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ELU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ELU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ELU(),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
        ]
        self.decoder = nn.Sequential(*module_list)
        

    def forward(self, slots):
        return self.decoder(slots)

class AdditiveDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder(in_dim=params["slot_dim"], out_channels = 3)

    def forward(self, slots):
        # z [B, N, L, d]
        B, N, d = slots.shape
        slots = slots.reshape(B*N, d)
        recons = self.decoder(slots) 
        recons = recons.view(B, N, recons.shape[1:])
        img = recons.sum(1)        
        return img


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
        self.stage = stage
        assert self.stage in ["train", "eval"], "stage must be either 'train' or 'eval'"
        self.L = nlatents
        self.A = A

        self.encoder_cnn = Encoder(self.resolution)
        self.hid_dim = self.encoder_cnn.last_dim
        self.ldim = latent_dim
        self.mlp = nn.Sequential(
             nn.Linear(self.hid_dim, self.hid_dim), # used to be hid_dim, hid_dim
             nn.ELU(),
             nn.Linear(self.hid_dim, self.hid_dim)
        )
        self.slot_attention = SlotAttention(
                num_slots=self.num_slots,
                slot_dim = self.slot_dim,
                dim=self.hid_dim,
                iters = self.num_iterations,
                eps = 1e-8, 
                hidden_dim = 128)
        self.noise_nets = nn.ModuleDict()
        for var_name in params["latent_vars"]:
            self.noise_nets[var_name] = VAE(nin = self.ldim,
                                            nh = self.ldim,
                                            nout = self.ldim
                                            )
        
        self.decoder = AdditiveDecoder()
        self.causal_flow = MultivariateCausalFlow(self.L, self.ldim, self.A)
        self.prior = PriorMultivariateCausalFlow(self.L, self.ldim, self.A)
        self.mmd_loss = MMDLoss()
        if params["TC_minimization"]: self.tc_discriminator = TCDiscriminator(self.L*self.ldim, self.L*self.ldim*4)
        

        self.some_labels = torch.tensor([])
        self.gmm_fit_epoch = 20 if params["alpha"] > 0. else 0. # 200
        self.epoch = 0
        
    
    def forward(self, img, labels):

        B, _, _, _ = img.shape
        n_s = params['num_slots'] 

        assert labels.shape == (B, n_s, self.L), f"{labels.shape}"

        self.img = img
        x = self.encoder_cnn(self.img)
        x = nn.LayerNorm(x.shape[1:]).to(device)(x)
        self.features_to_slots = self.mlp(x)
        slots, self.slot_pos, attn = self.slot_attention(self.features_to_slots, num_slots=n_s) # slots in [B, n_s, d]
        self.attn = attn
            
        if False:
        #if self.is_train and self.step % params['step_size_img_sv'] == 0:
            aux_attn = attn.reshape((B, n_s, 32, 32))
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
        for var in params["latent_vars"]:
            mu, logvars, e = self.noise_nets[var].encode(slots)
            z_noise.append(e)
            mu_noise.append(mu)
            logvar_noise.append(logvars) 
        z_noise = torch.stack(z_noise).permute(1, 2, 0, 3)            # [B, N, L, d] => 'L' refers to the number of categorical latents
        mu_noise = torch.stack(mu_noise).permute(1, 2, 0, 3)          # [B, N, L, d]
        logvar_noise = torch.stack(logvar_noise).permute(1, 2, 0, 3)  # [B, N, L, d]
        assert z_noise.size(2) == self.L
        
        u_m = torch.ones(B, n_s, self.L, self.ldim)
        for i, var in enumerate(params["latent_vars"]):
            u_m[:, :, i] = ((labels[:, :, i] - params["factor_ranges"][var][0]) / (params["factor_ranges"][var][1] - params["factor_ranges"][var][0])).unsqueeze(-1).expand(-1, -1, self.ldim)
        
        if params["causal_dropout_prob"] <= 1.:
            z_scm, self.L_slopes, ss, ts = self.causal_flow(z_noise)
            self.ss, self.ts = ss, ts
        else:
            z_scm = z_noise 

        z_noise = z_noise.reshape(B*n_s, self.L, self.ldim)
        mu_noise = mu_noise.reshape(B*n_s, self.L, self.ldim)
        logvar_noise = logvar_noise.reshape(B*n_s, self.L, self.ldim)
        self.logvar_noise = logvar_noise

        
        slot_hat = []
        for i, var in enumerate(params["latent_vars"]):
            slot_hat.append(self.noise_nets[var].decode(z_scm[:, :, i, :]))
        slot_hat = torch.sum(torch.stack(slot_hat, dim=-2), dim=-2)
        
        img_hat = self.decoder(slot_hat)

        """
        - prior won't be aligned with labels...
        """

        if params["beta"] > 0:
            cp, self.prior_L_slopes, prior_ss, prior_ts = self.prior(u_m, z_scm) # [B, N, L, d]
            self.prior_ss, self.prior_ts = prior_ss, prior_ts     
                            
        if self.stage == "eval": return z_noise.reshape(B, n_s, self.L, self.ldim)
        return slots, img_hat, z_noise, mu_noise, logvar_noise, z_scm, slot_hat, cp
        
        
        
        
        
        

