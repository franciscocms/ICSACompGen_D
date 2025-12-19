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


# add project path to sys to import relative modules
import sys
import os
sys.path.insert(0, '../..') # add parent-parent dir

from src.main import params

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
    return np.exp((step-total_steps)*0.1)*final

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





class Encoder(nn.Module):
    def __init__(self, nin, nh, nout):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(nin, nh),
            nn.ELU(),
            nn.Linear(nh, nh),
            nn.ELU(),
        )
        self.means = nn.Sequential(nn.Linear(nh, nout), nn.Identity())
        self.logvars = nn.Linear(nh, nout)
    
    def forward(self, x):
        x = self.layers(x)
        return self.means(x), self.logvars(x)

class Decoder(nn.Module):
    def __init__(self, nin, nh, nout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(nin, nh),
            nn.ELU(),
            nn.Linear(nh, nh),
            nn.ELU(),
            nn.Linear(nh, nout),
            #nn.Sigmoid()
        )
    def forward(self, z):
        return self.layers(z)

class VAE(nn.Module):
    def __init__(self, nin, nh, nout):
        super().__init__()
        self.encoder = Encoder(nin, nh, nout)                # takes an embedding batch u_m 
        self.decoder = Decoder(nin=nout, nh=nh, nout=nin)    # takes the reparameterized z to generate hat(u_m)

    def compute_kl_loss(self, mu, logvars):
        kl_div = -0.5 * torch.sum(1 + logvars - mu**2 - torch.exp(logvars), dim=-1)
        return kl_div
    
    def compute_rec_loss(self, rec, target):
        criterion = nn.MSELoss()
        return criterion(rec, target)

    def encode(self, x):
        mu, logvars = self.encoder(x)
        assert not torch.isnan(logvars).any()
        stds = torch.exp(0.5*logvars)
        z = mu + torch.randn_like(mu)*stds # reparameterization trick
        return mu, logvars, z
    
    def decode(self, z):
        rec = self.decoder(z)
        return rec

class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh, f_act="identity"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.ELU(),
            nn.Linear(nh, nh),
            nn.ELU(),
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
    def __init__(self, L, ldim, A=None, net_class=MLP, nh=100, f_act="identity"):
        super().__init__()

        self.s_cond = net_class(L*ldim, ldim, nh, f_act=f_act)
        self.t_cond = net_class(L*ldim, ldim, nh, f_act=f_act)
        self.total_dims = L*ldim
        
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
        ss = torch.tensor([]).to(device)
        ts = torch.tensor([]).to(device)

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
            self.nf[str(i)] = Flow(L, ldim, A, f_act="tanh") # instead of "tanh"
        

    
    def forward(self, u_m, z):

        # e:      [B, N, L, d]  ->  label embeddings
        # latent: [B, N, L, d]  ->  z_scm
        """ r [B*N, L, K] are the responsibilities to guarantee identifiability """

        B, N, L, d = z.shape

        assert z.shape == u_m.shape

    
        cp_m = torch.zeros_like(u_m).to(device)
        L_slopes = torch.tensor([]).to(device)
        ss = torch.tensor([])
        ts = torch.tensor([])

        total_dims = self.L*self.ldim
        for i in range(self.L):
            if 1 in self.A[i, :]: # shape
                mask = self.A[i, :].repeat(self.ldim, 1).T.reshape(total_dims).to(device)
            else: # color
                mask = torch.zeros(total_dims).to(device)
            
            cp_m[:, :, i, :], s, t = self.nf[str(i)](u_m[:, :, i, :], z, mask)
            L_slopes = torch.cat((L_slopes, s.unsqueeze(-2)), dim=-2)
            
            ss = torch.cat((ss, s.unsqueeze(-2)), dim=-2)
            ts = torch.cat((ts, t.unsqueeze(-2)), dim=-2)

        return cp_m, L_slopes, ss, ts


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