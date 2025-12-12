from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch import Tensor
from torch.nn import functional as F
from einops import reduce
import math
from scipy.optimize import linear_sum_assignment

import pyro.poutine as poutine
from pyro.poutine.runtime import apply_stack
from pyro.infer import TracePosterior
import pyro.distributions as dist

# add project path to sys to import relative modules
import sys
import os
sys.path.append(os.path.abspath(__file__+'/../../'))

from main import setup
from .distributions import CategoricalVals, TruncatedNormal
from .var import Variable

params = setup.params
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging
logger = logging.getLogger(params["running_type"])


def linear_scheduler(step, total_steps, initial, final):
    """Linear scheduler"""

    if step >= total_steps:
        return final
    if step <= 0:
        return initial
    if total_steps <= 1:
        return final

    t = step / (total_steps - 1)
    return (1.0 - t) * initial + t * final

def exp_scheduler(step, total_steps, initial, final):
    """Exponential scheduler"""

    if step >= total_steps:
        return final
    if step <= 0:
        return initial
    if total_steps <= 1:
        return final

    t = step / (total_steps - 1)
    return np.exp((step-total_steps)*0.005)*final

def exp_scheduler_neg(step, total_steps, initial, final):
    """Exponential scheduler"""

    if step >= total_steps:
        return final
    if step <= 0:
        return initial
    if total_steps <= 1:
        return final

    t = step / (total_steps - 1)
    return np.exp((1/(step-2020))*20)

def to_int(value: Tensor):
    return int(torch.round(value))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def kl_normal(qm, qv, pm, pv):
    
    # logger.info(qm.shape)
    # logger.info(qv.shape)
    # logger.info(pm.shape)
    # logger.info(pv.shape)

    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    if len(element_wise.shape) == 2:
        kl = torch.sum(element_wise, dim=-1)
    else:
        # dim_to_reduce = tuple([d for d in kl.shape if d != 0])
        # kl = torch.sum(element_wise, dim=dim_to_reduce)
        
        #logger.info(element_wise.shape)
        
        #kl = torch.mean(torch.sum(element_wise, dim=(-1, -2)), dim=-1)
        kl = element_wise

	#print("log var1", qv)
    return kl



"""
MMD classes below from https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py
"""

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        assert not torch.isnan(L2_distances).any()

        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


class NoiseNet(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU()
        )
        self.fc_mean = nn.Linear(hid_dim, out_dim)
        self.fc_var = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = self.fc(x)
        mean = self.fc_mean(x)
        logvar = self.fc_var(x)
        assert not torch.isnan(logvar).any()

        var = torch.exp(logvar)
        return mean, var


class MixtureEncoder(nn.Module):
    def __init__(self, nin, nh, nout):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(nin, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU()
        )
        self.means = nn.Sequential(nn.Linear(nh, nout), nn.Identity())
        self.logvars = nn.Linear(nh, nout)
    
    def forward(self, x):
        x = self.layers(x)
        return self.means(x), self.logvars(x)

class MixtureDecoder(nn.Module):
    def __init__(self, nin, nh, nout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(nin, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, nout),
            #nn.Sigmoid()
        )
    def forward(self, z):
        return self.layers(z)

class MixtureVAE(nn.Module):
    def __init__(self, nin, nh, nout):
        super().__init__()

        self.encoder = MixtureEncoder(nin, nh, nout)                # takes an embedding batch u_m 
        self.decoder = MixtureDecoder(nin=nout, nh=nh, nout=nin)    # takes the reparameterized z to generate hat(u_m)

    def forward(self, x):
        mu, logvars = self.encoder(x)
        assert not torch.isnan(logvars).any()

        stds = torch.exp(0.5*logvars)
        z = mu + torch.randn_like(mu)*stds # reparameterization trick
        rec = self.decoder(z)
        return rec, mu, logvars, z

class MixturePrior(nn.Module):
    def __init__(self, label, num_classes_list, nin = 4, nh = 128, ldim=4, step_size=1):
        super().__init__()

        self.label = label
        n_latents = len(num_classes_list)
        self.ldim = ldim

        assert all(num_classes_list[0] == x for x in num_classes_list)
        self.K = num_classes_list[0]

        self.prior_mixt = MixtureVAE(nin, nh, self.ldim)
        
        # self.pi_c = nn.Parameter(torch.ones(n_latents, self.K)*0.5)
        # self.mu_c = nn.Parameter(torch.randn(n_latents, self.K, nin))
        # self.logvar_c = nn.Parameter(torch.ones(n_latents, self.K, nin)*-1)
        self.pi_c = torch.ones(self.K)/self.K
        self.mu_c = torch.randn(self.K, self.ldim)
        self.logvar_c = torch.ones(self.K) * -1.


        self.step = 0
        self.step_size = step_size
        assert self.step_size > 0
        self.beta = 1.


    def forward(self, x):
        # x [B, L, d]
        B, L, d = x.shape

        # run the VaDE module correspondent to latent variable l
        x_hat, mu, logvars, z = self.prior_mixt(x)
        # x_ is [B, N, d]
        # mu is [B, N, ldim]
        # z is  [B, N, ldim]     
        
        return mu, logvars, x_hat, z
    
    def compute_yc(self, z, print_mixt_params=False, printed=False, yc_temperature=1.):
    
        # log q(c|x) = log p(c) + log p(z|c)
        log_pzc = self.gaussian_pdfs_log(z, self.mu_c, self.logvar_c)  
        if print_mixt_params and not printed:
            logger.info(f"log p(e|c): {log_pzc[:5]}")

        log_prob = torch.log(self.pi_c + 1e-10) + log_pzc # [B, K]
        if print_mixt_params and not printed:
            logger.info(f"log prob before temperature: {torch.mean(log_prob, dim=0).tolist()}")
        log_prob = log_prob / yc_temperature
        if print_mixt_params and not printed: 
            logger.info(f"log prob after temperature: {torch.mean(log_prob, dim=0).tolist()}")
        
        log_prob = log_prob - torch.logsumexp(log_prob, dim=-1, keepdim=True)  # normalize
        # if print_mixt_params:
        #     logger.info(f"log prob after normalization: {torch.mean(log_prob, dim=0).tolist()}")
        
        assert not torch.isnan(log_prob).any()

        yc = torch.exp(log_prob)  # [B, K]

        # if print_mixt_params:
        #     logger.info(f"log p(z|c) shape: {log_pzc.shape}") #[B*N, K]
        #     logger.info(f"log p(z|c): {log_pzc[:2]}")
        
        
        return yc

    def compute_elbo(self, mu, logvars, z, print_mixt_params, printed, yc_temperature):
        
        # aux_v = self.logvar_c + logvars/self.logvar_c + torch.square(mu - self.mu_c)/self.logvar_c
        # print(aux_v.shape) # [B, K, d]

        yc = self.compute_yc(z, print_mixt_params, printed, yc_temperature)
        #rec_loss = self.compute_rec_loss(x_hat, x)

        B, num_comp = yc.shape

        # logger.info(yc.shape)
        # logger.info(self.logvar_c.shape)
        # logger.info(logvars.shape)
        # logger.info(mu.shape)
        # logger.info(self.mu_c.shape) 
        
        # yc = torch.clamp(yc, 1e-3, 1.)
       
        # loss in j135
        # mixt_kl = 0.5 * torch.sum(yc * torch.mean(self.logvar_c + torch.exp(logvars)/torch.exp(self.logvar_c) + torch.square(mu - self.mu_c)/torch.exp(self.logvar_c), dim=-1), dim=-1)
        #kl_reg = torch.sum(yc * torch.log(self.pi_c/yc), dim=-1) + 0.5*torch.mean(1 + logvars, dim=-1)

        mixt_kl = 0.5 * torch.sum(yc * torch.sum(self.logvar_c.unsqueeze(-1).repeat(1, self.ldim) + torch.exp(logvars)/torch.exp(self.logvar_c.unsqueeze(-1).repeat(1, self.ldim)) + torch.square(mu - self.mu_c)/torch.exp(self.logvar_c.unsqueeze(-1).repeat(1, self.ldim)), dim=-1), dim=-1)
        #kl_reg = torch.sum(yc * torch.log(self.pi_c/yc), dim=-1) 
        #kl_reg = torch.sum(yc * torch.log(yc), dim=-1)
        
        
        uniform = torch.full_like(torch.mean(yc, dim=0), 1.0 / num_comp)
        kl_uni = (torch.mean(yc, dim=0) * (torch.log(torch.mean(yc, dim=0) + 1e-12) - torch.log(uniform))).sum()        
        kl_reg = -torch.mean(torch.sum(yc * torch.log(yc + 1e-12), dim=-1)) + kl_uni
        
        
        log_var_reg = 0.5*torch.sum(1 + logvars, dim=-1)
        #log_var_reg = 0.5*torch.sum(0.5 + logvars, dim=-1)
        #log_var_reg = 0.5*torch.sum(0.1 + logvars, dim=-1)


        mixt_kl = torch.mean(mixt_kl) #* 0.01
        #kl_reg = torch.mean(kl_reg)
        log_var_reg = torch.mean(log_var_reg)
        
        if print_mixt_params and not printed:
            logger.info(f"mixt kl loss: {mixt_kl}")
            #logger.info(f"(mu - mu_c) shape: {torch.square(mu - self.mu_c).shape}") # [B*N, L, d]
            logger.info(f"(mu - mu_c): {torch.sum(torch.square(mu - self.mu_c), dim=-1)[:5]}")
            logger.info(f"yc * (mu - mu_c): {(yc * torch.sum(torch.square(mu - self.mu_c), dim=-1))[:5]}")
            #logger.info(f"(mu - mu_c)/var_c: {torch.mean(torch.sum(torch.sum(torch.square(mu - self.mu_c)/torch.exp(self.logvar_c), dim=-1), dim=-1))}")

            logger.info(f"kl uni: {kl_uni}")
            logger.info(f"kl reg: {kl_reg}")
            logger.info(f"logvar reg: {log_var_reg}")

            # logger.info(f"\nmu: {mu}")
            # logger.info(f"mu_c: {self.mu_c}")
            # logger.info(f"mu-mu_c: {torch.square(mu-self.mu_c)}")
            # logger.info(f"(mu-mu_c)/exp(logvar_c): {torch.square(mu-self.mu_c)/torch.exp(self.logvar_c)}\n")
        
        return mixt_kl, kl_reg, log_var_reg, yc
    
    def compute_rec_loss(self, rec, target):
        criterion = nn.MSELoss()
        return criterion(rec, target)
    
    def gaussian_pdfs_log(self, x , mus, log_sigma2s):
        """
        x:              [B, d]
        mus:            [K, d]
        log_sigma2s:    [K]
        """


        G=[]
        for c in range(self.K):
            G.append(self.gaussian_pdf_log(x, mus[c:c+1, :], log_sigma2s[c:c+1]).view(-1,1))
        return torch.cat(G,1)

    def gaussian_pdf_log(self, x, mu, log_sigma2):
        assert not torch.isnan(log_sigma2).any()

        return -0.5*(torch.sum(math.log(math.pi*2) + log_sigma2 + torch.square(x-mu)/torch.exp(log_sigma2), 1))
    
    def kl_div_mixt(self, logvar, prior_logvar):
        B, d = logvar.shape
        K = prior_logvar.shape[0] # [K, d]

        # Expand for broadcasting
        logvar = logvar[:, None, :]         # [B, 1, d]
        prior_logvar = prior_logvar[None, :, :]  # [1, K, d]

        # Compute KL divergence for each (B, K) pair
        kl = 0.5 * (prior_logvar - logvar + torch.exp(logvar - prior_logvar) - 1)
        return kl.sum(-1)  # sum over feature dimension -> [B, K]


class LabelEmbeddings(nn.Module):
  def __init__(self, num_classes_list, latent_dim):
    """
    num_classes_list: List of number of categories for each discrete variable (e.g. [2, 3] for shape and color)
    latent_dim: Dimensionality of the latent space
    """
    super().__init__()
    self.embeddings = nn.ModuleList([
        nn.Sequential(nn.Embedding(num_classes, latent_dim), nn.Identity()) for num_classes in num_classes_list
    ])
    self.latent_dim = latent_dim
    self.eps = 1e-1

  def forward(self, labels):
    """
    labels: Tensor of shape [batch_size, num_discrete_variables] with categorical indices
    Returns:
        mean, var: Prior mean and variance for each latent variable
    """

    #logger.info(labels.shape)

    batch_size, num_vars = labels.shape
    #vars = torch.ones(batch_size, 1, self.latent_dim).to(labels.device)  # Keep variance constant
    means = torch.tensor([])
    for j in range(num_vars): 
        embedding = self.embeddings[j](labels[:, j]).unsqueeze(-2) + torch.randn(batch_size, 1, self.latent_dim) * params["label_embeddings_std"]
        means = torch.cat((means, embedding), dim=-2) # [B, L, d]
        
    return means#, vars # [B, L, d]
  
class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh, f_act="identity"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nout),
        )
        if f_act == "softmax": self.f_act = nn.Softmax(dim=-1)
        elif f_act == "sigmoid": self.f_act = nn.Sigmoid()
        elif f_act == "tanh": self.f_act = nn.Tanh()
        elif f_act == "identity": self.f_act = nn.Identity()
        elif f_act == "softplus": self.f_act = nn.Softplus()

    def forward(self, x, mask=None):
        if mask == None: 
            return self.f_act(self.net(x))
        else:
            return self.f_act(self.net(x * mask))
        
class ShallowMLP(nn.Module):
    def __init__(self, nin, nout, nh, f_act="identity"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nout)
        )
        if f_act == "softmax": self.f_act = nn.Softmax(dim=-1)
        elif f_act == "sigmoid": self.f_act = nn.Sigmoid()
        elif f_act == "tanh": self.f_act = nn.Tanh()
        else: self.f_act = nn.Identity()

    def forward(self, x):
        return self.f_act(self.net(x))

class LinearProbe(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nin),
            nn.ReLU(),
            nn.Linear(nin, nout)
        )

    def forward(self, x):
        return self.net(x)

class Flow(nn.Module):
    def __init__(self, L, ldim, A=None, net_class=MLP, nh=100, prior=False, f_act="identity"):
        super().__init__()

        if not prior: 
            self.s_cond = net_class(L*ldim, ldim, nh, f_act=f_act)
            self.t_cond = net_class(L*ldim, ldim, nh, f_act=f_act)
            self.total_dims = L*ldim
        else:
            self.s_cond = net_class(L*2*ldim, ldim, nh, f_act=f_act)
            self.t_cond = net_class(L*2*ldim, ldim, nh, f_act=f_act)
            self.total_dims = L*2*ldim
        self.prior = prior
        
        self.L, self.ldim = L, ldim

        self.alpha = 0.1

    def forward(self, eps, z, mask):
        B, n_s, _ = eps.shape

        s = self.s_cond(z.reshape(-1, self.total_dims), mask).reshape(B, n_s, self.ldim) # slope
        t = self.t_cond(z.reshape(-1, self.total_dims), mask).reshape(B, n_s, self.ldim) # offset
        
        assert not torch.isnan(s).any()
        assert not torch.isnan(t).any()
        
        z_i = torch.exp(s) * eps + t

        return z_i, s, t
        
class MultivariateCausalFlow(nn.Module):
    def __init__(self, L, ldim, A):
        super().__init__()
        
        self.L, self.ldim, self.A = L, ldim, A
        self.nf = nn.ModuleDict()
        for i in range(L):
            self.nf[str(i)] = Flow(L, ldim, A, f_act="tanh") # instead of "tanh"

    def forward(self, eps):
        z = torch.zeros_like(eps).to(device)
        L_slopes = torch.tensor([]).to(device)
        ss = torch.tensor([])
        ts = torch.tensor([])

        total_dims = self.L*self.ldim
        for i in range(self.L):
            if 1 in self.A[i, :]:
                mask = self.A[i, :].repeat(self.ldim, 1).T.reshape(total_dims).to(device)                        
            else:
                mask = torch.zeros(total_dims).to(device)
            
            z[:, :, i, :], s, t = self.nf[str(i)](eps[:, :, i, :], z, mask)
            L_slopes = torch.cat((L_slopes, s.unsqueeze(-2)), dim=-2)
            
            ss = torch.cat((ss, s.unsqueeze(-2)), dim=-2)
            ts = torch.cat((ts, t.unsqueeze(-2)), dim=-2)

        return z, L_slopes, ss, ts

class PriorMultivariateCausalFlow(nn.Module):
    def __init__(self, L, ldim, A):
        super().__init__()
        
        self.L, self.ldim, self.A = L, ldim, A
        self.nf = nn.ModuleDict()
        for i in range(L):
            self.nf[str(i)] = Flow(L, ldim, A, prior=True, f_act="tanh") # instead of "tanh"
        
        self.embeddings = nn.ModuleList([nn.Embedding(3, ldim) for _ in range(self.L)])

    
    def forward(self, u_m, z=None, gt_labels=None):

        # e:      [B, N, L, d]  ->  label embeddings
        # latent: [B, N, L, d]  ->  z_scm
        """ r [B*N, L, K] are the responsibilities to guarantee identifiability """

        B, N, L, d = z.shape

        assert z.shape == u_m.shape

        """
        - the label embeddings have preserved the object order that is initially sampled
        - z_scm have an object order that comes from the arbitrary structure of slots (slots -> z_noise -> z_scm)
        - how to align both??? or at least, account for the disalignment like in a matching loss?
        - 
        """

        # assert preds != None

        # probs_exp = preds[:, :, None, :, :]  # [B, N, 1, L, K]
        # labels_exp = gt_labels[:, None, :, :]                # [B, 1, N, L]
        # labels_exp = labels_exp.expand(-1, N, -1, -1)      # [B, N, N, L]
        # labels_exp = labels_exp.unsqueeze(-1)             # [B, N, N, L, 1]

        # # Step 2: Compute log-probs
        # log_probs = (probs_exp + 1e-9).log()         # [B, N, 1, L, K]
        # log_probs = log_probs.expand(-1, -1, N, -1, -1)   # [B, N, N, L, K]

        # # Step 3: Gather NLL loss at GT label positions
        # nll = -log_probs.gather(dim=-1, index=labels_exp).squeeze(-1)  # [B, N, N, L]

        # # Step 4: Aggregate across latent variables (mean or sum)
        # class_cost = nll.mean(dim=-1)  # Final cost: [B, N, N]

        # # if is_train and step % params["step_size"] == 0:
        # #     logger.info(f"\npreds: {preds[:5]}")
        # #     logger.info(f"labels: {gt_labels[:5]}")
        # #     logger.info(f"cost: {class_cost}")
        
        # a, b = torch.tensor([1.]), torch.tensor([1.])
        # #attn_logits = torch.cdist(e.reshape(B, N, L*d), latent.reshape(B, N, L*d)) 
        # #attn_logits = torch.cdist(e.reshape(B, N, L*d), latent.reshape(B, N, L*d)) + class_cost
        # attn_logits = class_cost
        # # logger.info(attn_logits.shape)    [B, N, N]   
        # attn_logits, p, q = minimize_entropy_of_sinkhorn(attn_logits, a, b, mesh_lr=3, n_mesh_iters=3) 
        # # logger.info(attn_logits.shape)    [B, N, N]
        # attn, _, _ = sinkhorn(attn_logits, a, b, u=p, v=q)
        # # logger.info(attn.shape)           [B, N, N]

        # # if is_train and step % params["step_size"] == 0:
        # #     logger.info(f"align attn: {attn[:5]}")
        
        # u_m = torch.matmul(attn, u_m.reshape(B, N, L*d)).reshape(B, N, L, d)
    
        cp_m = torch.zeros_like(u_m).to(device)
        L_slopes = torch.tensor([]).to(device)
        ss = torch.tensor([])
        ts = torch.tensor([])

        self.embeddings_stack = torch.tensor([])
        for j in range(L): 
            embedding = self.embeddings[j](gt_labels[:, :, j].long()).unsqueeze(-2)
            self.embeddings_stack = torch.cat((self.embeddings_stack, embedding), dim=-2) # [B, N, L, d]

        #self.embeddings_stack = torch.matmul(attn, self.embeddings_stack.reshape(B, N, L*d)).reshape(B, N, L, d)
        z = torch.cat((z, self.embeddings_stack), dim=-1) # [B, N, L, d+d]
        
        """
        I need to condition on z and on the responsibilities
        - take gt_labels [B, N, L] and transform to embeddings for [B, N, d]
        - different set of embeddings for different variables!!!!
        - align the embeddings just like I align u_m above
        - concat these embeddings on z, so that each element in z has the class embedding 
        """

        total_dims = self.L*self.ldim
        for i in range(self.L):
            if 1 in self.A[i, :]: # shape
                mask = torch.cat((self.A[i, :].repeat(self.ldim, 1).T.reshape(total_dims).to(device), (1. - self.A[i, :].repeat(self.ldim, 1).T.reshape(total_dims)).to(device)), dim=0)
            else: # color
                mask = torch.cat((torch.zeros(total_dims).to(device), torch.ones(total_dims//2).to(device), torch.zeros(total_dims//2).to(device)), dim=0)
            
            # logger.info(f"var {i} mask: {mask.shape} - {mask}")
            # logger.info(f"z: {z.shape} - {z}")
            
            cp_m[:, :, i, :], s, t = self.nf[str(i)](u_m[:, :, i, :], z, mask)
            L_slopes = torch.cat((L_slopes, s.unsqueeze(-2)), dim=-2)
            
            ss = torch.cat((ss, s.unsqueeze(-2)), dim=-2)
            ts = torch.cat((ts, t.unsqueeze(-2)), dim=-2)

        return cp_m, L_slopes, ss, ts

def loft(s, tau = params["slope_constraint"]):
    tau = torch.tensor(tau, dtype=s.dtype, device=device)
    return torch.sign(s) * (torch.log(torch.maximum(torch.abs(s) - tau, torch.tensor(0.)) + 1) + torch.minimum(torch.abs(s), tau))

def old_compute_leakage_loss(logits, labels):
    B, N, K = logits.shape
    loss = 0.0
    for b in range(B):
        cost_matrix = torch.zeros(N, N)
        for i in range(N):  # predicted slot
            for j in range(N):  # label slot
                # Compute CE between predicted logit and label[j]
                ce = F.cross_entropy(logits[b, i], labels[b, j], reduction='none')
                cost_matrix[i, j] = ce
        
        cost_matrix_np = cost_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix_np)

        matched_logits = logits[b, row_ind]           # [N, K]
        matched_labels = labels[b, col_ind]           # [N]
        ce_loss = F.cross_entropy(
            matched_logits,
            matched_labels,
            reduction='sum'
        )
        loss += ce_loss

    loss = loss / B
    return loss  # negative CE = leakage loss to maximize for disentanglement

def compute_leakage_loss(logits, labels):
    B, N, K = logits.shape
    loss = 0.0
    logits_exp = logits.unsqueeze(2).expand(-1, -1, N, -1)       # [B, N, N, K]
    labels_exp = labels.unsqueeze(1).expand(-1, N, -1)           # [B, N, N]

    # Compute cross-entropy per element: flatten to batch for CE
    ce_loss_matrix = F.cross_entropy(
        logits_exp.reshape(-1, K),           # [B*N*N, K]
        labels_exp.reshape(-1),              # [B*N*N]
        reduction='none'
    ).reshape(B, N, N)  # [B, N, N]

    for b in range(B):
        cost = ce_loss_matrix[b].detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        
        matched_logits = logits[b, row_ind]        # [N, K]
        matched_labels = labels[b, col_ind]        # [N]
        ce = F.cross_entropy(matched_logits, matched_labels, reduction='sum')
        loss += ce
    
    loss = loss / B
    return loss  # negative CE = leakage loss to maximize for disentanglement



class TCDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, 2),
            #nn.LeakyReLU(0.2, True),
            #nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x).squeeze()

def permute_dims(z, d):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = torch.tensor([])
    perms = []
    for z_j in z.split(d, 1):
        #print(z_j.shape)
        perm = torch.randperm(B).to(z.device)
        perms.append(perm)
        perm_z_j = z_j[perm]
        perm_z = torch.cat((perm_z, perm_z_j), dim=-1)

    return perm_z, perms






class GaussianNet(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim, activ):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU()
        )
        self.fc_mean = nn.Linear(hid_dim, out_dim)
        self.fc_logvar = nn.Linear(hid_dim, out_dim)
        self.activ = activ

    def forward(self, x):
        x = self.fc(x)
        mean = self.activ(self.fc_mean(x))
        logvar = self.fc_logvar(x)
        return mean, logvar


def get_pretrained_wts(guide, path):
    # loading the pre-trained weights from the density map estimator
    pretrained_estimator = torch.load(path)
    guide_encoder_dict = guide.encoder_cnn.state_dict()
    for key in guide_encoder_dict:
        if key in pretrained_estimator.keys():
            guide_encoder_dict[key] = pretrained_estimator[key]
            logging.info(f"...{key} loaded")
    guide.encoder_cnn.load_state_dict(guide_encoder_dict)
    return guide

@torch.jit.script
def sinkhorn(C: Tensor, a: Tensor, b: Tensor, n_sh_iters: int = 5, temperature: float = 1, u: Union[Tensor, None] = None, v: Union[Tensor, None] = None) -> Tuple[Tensor, Tensor, Tensor]:
    p = -C / temperature
    log_a = torch.log(a)
    log_b = torch.log(b)

    # C is [B, num_inputs, num_slots]

    if u is None:
        u = torch.zeros_like(a)
    if v is None:
        v = torch.zeros_like(b)

    for _ in range(n_sh_iters):
        u = log_a - torch.logsumexp(p + v.unsqueeze(1), dim=2) # rows normalization, logsumexp occurs over slots dimension -> normalize slots for each feature
        v = log_b - torch.logsumexp(p + u.unsqueeze(2), dim=1) # cols normalization, logsumexp occurs over features dimension -> normalize features for each slot

    logT = p + u.unsqueeze(2) + v.unsqueeze(1)
    return logT.exp(), u, v


@torch.enable_grad()
def minimize_entropy_of_sinkhorn(C_0, a, b, noise=None, mesh_lr=1, n_mesh_iters=4, n_sh_iters=5, reuse_u_v=True):
    if noise is None:
        noise = torch.randn_like(C_0)

    C_t = C_0 + params["mesh_noise"] * noise 
    C_t.requires_grad_(True)

    u = None
    v = None
    for i in range(n_mesh_iters):
        attn, u, v = sinkhorn(C_t, a, b, u=u, v=v, n_sh_iters=n_sh_iters)

        if not reuse_u_v:
            u = v = None

        entropy = reduce(
            torch.special.entr(attn.clamp(min=1e-20, max=1)), "n a b -> n", "mean"
        ).sum()
        
        #logging.info(entropy)
        
        (grad,) = torch.autograd.grad(entropy, C_t, retain_graph=True)
        grad = F.normalize(grad + 1e-20, dim=[1, 2])
        C_t = C_t - mesh_lr * grad

    # attn, u, v = sinkhorn(C_t, a, b, u=u, v=v, num_sink=num_sink_iters)

    if not reuse_u_v:
        u = v = None

    return C_t, u, v

def assert_shape(actual: Union[torch.Size, Tuple[int, ...]], expected: Tuple[int, ...], message: str = ""):
    assert actual == expected, f"Expected shape: {expected} but passed shape: {actual}. {message}"

def linear_std(step):
    if step <= 1000: return 1. - step*0.00095
    else: return 0.05

def load_trained_guide(guide, GUIDE_PATH, stage=params["running_type"]):
    
    assert stage in ["train", "eval"]
    
    out_dim = 0
    pretrained_guide = torch.load(GUIDE_PATH) if params['device'] == torch.device('cuda:0') else torch.load(GUIDE_PATH, map_location='cpu')
    guide_dict = guide.state_dict()
    for k, v in pretrained_guide.items():
        
        #logger.info(k)
        
        if k not in guide_dict.keys():
            prior_distribution = None
            proposal_distribution = None
            name = k.split(".")[1]
            if name == 'mask':
                proposal_distribution = "bernoulli"
                out_dim = 1
            elif name in ["shape", "color", "size"]: 
                prior_distribution = "categorical"
                proposal_distribution = "categorical"
                out_dim = 3
            elif name in ["locX", "locY"]: 
                prior_distribution = "uniform"
                proposal_distribution = "normal"
                out_dim = 2

            # for now, instance will be set on 0!
            var = Variable(name=name,
                            value=None,
                            prior_distribution=prior_distribution,
                            proposal_distribution=proposal_distribution,
                            address=name.split("_")[0]
                            )
            
            #if prior_distribution is None: raise ValueError("Prior distribution cannot be None")
            #if proposal_distribution is None: raise ValueError("Proposal distribution cannot be None")
            
            if k.split("_")[0] == "prop": 
                if params["conditioned_classifier"]:
                    guide.add_proposal_net(var, out_dim, child=False)
                else:
                    if var.name == "shape":
                        guide.add_proposal_net(var, out_dim, child=True)
                    else:
                        guide.add_proposal_net(var, out_dim, child=False)

                
    guide_dict = guide.state_dict() 

    #logger.info(guide)
    
    if stage == "train":
        state_dict = {k: v for k, v in pretrained_guide.items() if k in guide_dict.keys() and k.split('.')[0] not in ["tc_discriminator", "prop_nets", "s_cond", "t_cond"]}
    elif stage == "eval":
        state_dict = {k: v for k, v in pretrained_guide.items() if k in guide_dict.keys() and k.split('.')[0] not in ["tc_discriminator"]}
    
    guide_dict.update(state_dict)
    guide.load_state_dict(guide_dict)
    #logging.info("ICSA guide successfully loaded...")
    return guide


from collections import deque

def topological_order_from_G(G):
    """
    G: (L,L) numpy array of 0/1 where G[i,j]=1 if z_j is parent of z_i (edge j -> i).
    Returns: list of node indices in a valid topological order (parents before children),
             or raises ValueError if the graph has a cycle.
    """
    #G = np.asarray(G, dtype=int)
    L = G.shape[0]
    if G.shape != (L, L):
        raise ValueError("G must be square")

    # compute in-degree for each node (number of parents)
    indeg = G.sum(axis=1)   # indeg[i] = sum_j G[i,j] = number of parents of node i

    # start with nodes that have zero in-degree (no parents)
    q = deque([i for i in range(L) if indeg[i] == 0])
    order = []

    # adjacency: edges j->i are indicated by G[i,j]==1
    # for convenience, compute children lists
    children = [[] for _ in range(L)]
    for i in range(L):
        for j in range(L):
            if G[i, j]:
                children[j].append(i)   # j -> i

    while q:
        u = q.popleft()
        order.append(u)
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) != L:
        raise ValueError("Graph contains a cycle; no topological ordering exists.")
    return order