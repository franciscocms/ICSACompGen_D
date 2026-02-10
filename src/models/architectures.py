import torch
import torch.nn as nn

from contextlib import nullcontext
from typing import Any, Dict

import sys
sys.path.insert(0, '../..') # add parent-parent dir

from src.main import params
from src.utils.guide import permute_dims

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging
logger = logging.getLogger(params["running_type"])

"""
Encoder and decoder architectures used in our AE model, the same as in https://arxiv.org/abs/2310.05327.

Wiedemer, T. et al. (2023).
Provable compositional generalization for object-centric learning.
"""

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        out = tensor.view(self.size)
        return out

def get_encoder(in_channels, out_dim):
    """
    Encoder f^{-1}: X -> Z; X - input image, Z - latent space.
    """
    modules_list = [
        nn.Conv2d(in_channels, 32, 4, 2, 1),
        nn.ELU(),
        nn.Conv2d(32, 32, 4, 2, 1),
        nn.ELU(),
        nn.Conv2d(32, 32, 4, 2, 1),
        nn.ELU(),
        nn.Conv2d(32, 32, 4, 2, 1),
        nn.ELU(),
        View((-1, 32 * 4 * 4)),
        nn.Linear(32 * 4 * 4, 256),
        nn.ELU(),
        nn.Linear(256, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, out_dim),
    ]

    encoder = nn.Sequential(*modules_list)
    return encoder

def get_decoder(in_dim, out_channels):
    """
    Decoder f: Z -> X; X - input image, Z - latent space.
    """
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
        nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
    ]

    decoder = nn.Sequential(*module_list)
    return decoder

class SlotMLPAdditive(torch.nn.Module):
    """
    Model predicts latent vector z_hat_i = f^{-1}(x) for each slot and generates x_hat = sum_{i=1}^{n_slots} f(z_hat_i),
    by summing the output of each slot. For the output z_hat reshaped into (batch_size, n_slots, n_slot_latents).
    Model outputs x_hat, z_hat and list of x_hat_i.
    """

    def __init__(
        self,
        in_channels: int,
        n_slots: int,
        n_slot_latents: int,
        no_overlap: bool = False,
    ) -> None:
        super(SlotMLPAdditive, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.no_overlap = no_overlap
        self.encoder = SlotEncoder(in_channels, n_slots, n_slot_latents)
        self.decoder = SlotMLPAdditiveDecoder(in_channels, n_slots, n_slot_latents)       

    def forward(
        self,
        x,
    ) -> Dict[str, Any]:
        """
        Compute forward pass of the model.
        Reconstruction: \hat{x} = sum_{i=1}^{n_slots} f(z_i)
        "Imagined" latent vectors: z_tilde = \pi(f^{-1}(\hat{x})})
        "Imagined" images: \hat{x_tilde} = sum_{i=1}^{n_slots} f(z_tilde_i)

        Args:
            x: input image, of shape (batch_size, in_channels, height, width)


        Returns:
            A tuple containing the following:
                - hat_x: reconstructed input image
                - hat_z: latent vectors for input image
                - figures: figures visualizing each latent vector
        """
        hat_z = self.encoder(x)
        hat_x, figures = self.decoder(hat_z)

        output_dict = {
            "reconstructed_image": hat_x,
            "predicted_latents": hat_z,
            "reconstructed_figures": figures.permute(1, 0, 2, 3, 4),
        }
        return output_dict

class SlotEncoder(torch.nn.Module):
    def __init__(self, in_channels, n_slots, n_slot_latents):
        super(SlotEncoder, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.encoder = get_encoder(in_channels, n_slots * n_slot_latents)

    def forward(self, x):
        x = self.encoder(x)
        out = x.view(-1, self.n_slots, self.n_slot_latents)
        return out

class SlotMLPAdditiveDecoder(torch.nn.Module):
    """
    Model generates x_hat = sum_{i=1}^{n_slots} f(z_i),
    by summing the output of each slot.  Model outputs x_hat and list of x_hat_i.
    """

    def __init__(
        self,
        in_channels: int,
        n_slots: int,
        n_slot_latents: int,
    ) -> None:
        super(SlotMLPAdditiveDecoder, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.decoder = get_decoder(n_slot_latents, in_channels)
        self.model_name = "SlotMLPAdditiveDecoder"

    def forward(self, latents):
        # Reshape latents from [batch_size, n_slots, features] to [batch_size*n_slots, features]
        batch_size, n_slots, features = latents.size()
        reshaped_latents = latents.view(batch_size * n_slots, features)

        # Pass reshaped latents through decoder
        reshaped_figures = self.decoder(reshaped_latents)

        # Reshape figures back to [batch_size, n_slots, ...]
        figures = reshaped_figures.view(
            batch_size, n_slots, *reshaped_figures.shape[1:]
        )
        # Sum over the n_slots dimension for image  
        image = figures.sum(dim=1)

        return image, figures
    
class Encoder(nn.Module):
    def __init__(self, nin, nh, nout):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(nin, nh),
            nn.ELU(),
            nn.Linear(nh, nh),
            nn.ELU(),
            nn.Linear(nh, nh),
            nn.ELU()
        )
        self.fc_mean = nn.Linear(nh, nout)
        self.fc_logvar = nn.Linear(nh, nout)

    def forward(self, x):
        x = self.layers(x)
        return self.fc_mean(x), self.fc_logvar(x)

class IdentityEncoder(nn.Module):
    def __init__(self, nin, nh, nout, slot_dim):
        super().__init__()
        # We define a dummy parameter so the optimizer doesn't crash 
        # if it expects parameters to optimize
        self.projection = nn.Linear(nin, nout)
    
    def forward(self, s):
        # s: [B, N, 64]
        mu = self.projection(s)
        
        # We set logvar to a very small number (e.g., -20) 
        # so variance is essentially zero
        logvar = torch.full_like(mu, -20.0) 
        
        return mu, logvar

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.net(x)
    
class Decoder(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        #self.proj = nn.Linear(nin, nout)
        nhidden = nin*4

        self.nets = nn.ModuleDict()
        for i in range(len(params["latent_vars"])):
            self.nets[str(i)] = nn.Sequential(nn.Linear(nin, nhidden),
                                         nn.ELU(), 
                                         nn.Linear(nhidden, nhidden),
                                         nn.ELU(),
                                         nn.Linear(nhidden, nhidden),
                                         nn.ELU(),
                                         nn.Linear(nhidden, nhidden),
                                         nn.ELU(),
                                         nn.Linear(nhidden, nout)
                                         )

    def forward(self, z):
        # z: [B, N, L, d]
        slot_hat = 0.
        for i in range(len(params["latent_vars"])):
            slot_hat += self.nets[str(i)](z[:, :, i])
        return slot_hat

class NoiseEncoder(nn.Module):
    def __init__(self, nin, nh, nout):
        super().__init__()
        self.encoder = Encoder(nin, nh, nout) 
        #self.encoder = IdentityEncoder(nin, nh, nout, slot_dim) 

    def encode(self, x):
        mu, logvars = self.encoder(x)
        stds = torch.exp(0.5*logvars)
        z = mu + torch.randn_like(mu)*stds # reparameterization trick
        return mu, logvars, z
    
class CausalDecoder(nn.Module):
    def __init__(self, nin, nout):  
        super().__init__()
        self.decoder = Decoder(nin=nin, nout=nout)    # takes the reparameterized z to generate hat(u_m)

    def forward(self, z):
        rec = self.decoder(z)
        return rec


class FullNoiseEncoder(nn.Module):
    def __init__(self, in_channels, n_slots, n_latents, latent_dim, stage="train"):
        super().__init__()

        self.stage = stage
        out_dim = n_slots * n_latents * latent_dim
        modules_list = [
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.ELU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ELU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ELU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ELU(),
            View((-1, 32 * 4 * 4)),
            nn.Linear(32 * 4 * 4, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            ]

        self.encoder = nn.Sequential(*modules_list)
        self.n_latents = n_latents
        self.latent_dim = latent_dim
        self.n_slots = n_slots
        self.fc_mean = nn.Linear(128, out_dim)
        self.fc_logvar = nn.Linear(128, out_dim)
    
    def encode(self, x):
        x = self.encoder(x)
        mean = self.fc_mean(x).view(-1, self.n_slots, self.n_latents*self.latent_dim)
        logvar = self.fc_logvar(x).view(-1, self.n_slots, self.n_latents*self.latent_dim)
        return mean, logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        stds = torch.exp(0.5*logvar)
        z = mu + torch.randn_like(mu)*stds # reparameterization trick
        return mu, logvar, z
        



class MLP(nn.Module):
    def __init__(self, nin, nout, nh, f_act="identity"):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(nin, nh),
        #     # nn.ELU(),
        #     # nn.Linear(nh, nh),
        #     nn.ELU(),
        #     nn.Linear(nh, nout),
        # )
        # if f_act == "softmax": self.f_act = nn.Softmax(dim=-1)
        # elif f_act == "sigmoid": self.f_act = nn.Sigmoid()
        # elif f_act == "tanh": self.f_act = nn.Tanh()
        # elif f_act == "identity": self.f_act = nn.Identity()
        # elif f_act == "softplus": self.f_act = nn.Softplus()

        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x, mask=None):
        return self.net(x)
        
class Predictor(nn.Module):
    def __init__(self, nin, nhidden, nout):
        super().__init__()
        
        nhidden = nin*4
        self.net = nn.Sequential(
            nn.Linear(nin, nhidden),
            nn.ELU(),
            nn.Linear(nhidden, nhidden),
            nn.ELU(),
            nn.Linear(nhidden, nout),
            #nn.Sigmoid()
        )
        #self.norm = nn.LayerNorm(nin)

    def forward(self, x):
        B, N, d = x.shape
        #x = self.norm(x)
        out = self.net(x)
        return out


class Flow(nn.Module):
    def __init__(self, L, ldim, A=None, net_class=MLP, nh=100, f_act="identity"):
        super().__init__()

        self.s_cond = net_class(L*ldim, ldim, ldim, f_act=f_act)
        self.t_cond = net_class(L*ldim, ldim, ldim, f_act="identity")
        self.total_dims = L*ldim
        
        self.L, self.ldim = L, ldim

    def forward(self, eps, z, mask):
        B, n_s, _ = eps.shape

        s = self.s_cond(z.reshape(-1, self.total_dims), mask).reshape(B, n_s, self.ldim) # slope
        t = self.t_cond(z.reshape(-1, self.total_dims), mask).reshape(B, n_s, self.ldim) # offset
        
        if torch.isnan(s).any():
            for name, param in self.s_cond.named_parameters():
                logger.info(f"{name} - {torch.isnan(param).any()}")

        assert not torch.isnan(s).any()
        assert not torch.isnan(t).any()

        # s = s - s.mean(dim=-1, keepdim=True)
        
        z_i = torch.exp(s) * eps + t
        return z_i, s, t
        
class MultivariateCausalFlow(nn.Module):
    def __init__(self, L, ldim, A):
        super().__init__()
        
        self.s_base = nn.Parameter(torch.randn(1, 1, ldim), requires_grad=True)
        self.t_base = nn.Parameter(torch.randn(1, 1, ldim), requires_grad=True)

        self.s2_base = nn.Parameter(torch.randn(1, 1, ldim), requires_grad=True)
        self.t2_base = nn.Parameter(torch.randn(1, 1, ldim), requires_grad=True)

        self.s3_base = nn.Parameter(torch.randn(1, 1, ldim), requires_grad=True)
        self.t3_base = nn.Parameter(torch.randn(1, 1, ldim), requires_grad=True)

        self.s_cond = MLP(ldim, ldim, ldim*4)
        self.t_cond = MLP(ldim, ldim, ldim*4)
        

    def forward(self, eps, epoch):        
        
        B, N, _, _ = eps.shape

        flag = torch.distributions.Categorical(torch.tensor([1/len(params["latent_vars"]) for _ in range(len(params["latent_vars"]))])
                                               ).sample()
        
        s0 = self.s_base.expand(B, N, -1)
        t0 = self.t_base.expand(B, N, -1)
        z0 = torch.exp(s0)*eps[:, :, 0, :] + t0

        s1 = self.s_cond(z0)
        t1 = self.t_cond(z0)
        z1 = torch.exp(s1)*eps[:, :, 1, :] + t1

        s2 = self.s2_base.expand(B, N, -1)
        t2 = self.t2_base.expand(B, N, -1)
        z2 = torch.exp(s2)*eps[:, :, 2, :] + t2

        s3 = self.s3_base.expand(B, N, -1)
        t3 = self.t3_base.expand(B, N, -1)
        z3 = torch.exp(s3)*eps[:, :, 3, :] + t3

        if epoch >= 100:
            # z1 is [B, N, d]
            with torch.no_grad():
                if flag == 0:
                    z0_int, _ = permute_dims(torch.flatten(z0, 0, 1), z0.size(-1))
                    z0_int = z0_int.view(B, N, -1)
                    
                    s1_int = self.s_cond(z0_int)
                    t1_int = self.t_cond(z0_int)
                    z1_int = torch.exp(s1_int)*eps[:, :, 1, :] + t1_int

                    self.z_int = torch.cat((z0_int.unsqueeze(-2),
                                            z1_int.unsqueeze(-2),
                                            z2.unsqueeze(-2),
                                            z3.unsqueeze(-2)), dim=-2)
                if flag == 1:
                    z1_int, _ = permute_dims(torch.flatten(z1, 0, 1), z1.size(-1))
                    z1_int = z1_int.view(B, N, -1)
                    self.z_int = torch.cat((z0.unsqueeze(-2),
                                            z1_int.unsqueeze(-2),
                                            z2.unsqueeze(-2),
                                            z3.unsqueeze(-2)), dim=-2)
                elif flag == 2:
                    z2_int, _ = permute_dims(torch.flatten(z2, 0, 1), z2.size(-1))
                    z2_int = z2_int.view(B, N, -1)
                    self.z_int = torch.cat((z0.unsqueeze(-2),
                                            z1.unsqueeze(-2),
                                            z2_int.unsqueeze(-2),
                                            z3.unsqueeze(-2)), dim=-2)
                
                elif flag == 3:
                    z3_int, _ = permute_dims(torch.flatten(z3, 0, 1), z3.size(-1))
                    z3_int = z3_int.view(B, N, -1)
                    self.z_int = torch.cat((z0.unsqueeze(-2),
                                            z1.unsqueeze(-2),
                                            z2.unsqueeze(-2),
                                            z3_int.unsqueeze(-2)), dim=-2)
        else:
            self.z_int = None

        z = torch.cat((z0.unsqueeze(-2),
                       z1.unsqueeze(-2),
                       z2.unsqueeze(-2),
                       z3.unsqueeze(-2)), dim=-2)
        ss = torch.cat((s0.unsqueeze(-2),
                        s1.unsqueeze(-2),
                        s2.unsqueeze(-2),
                        s3.unsqueeze(-2)), dim=-2)
        ts = torch.cat((t0.unsqueeze(-2),
                        t1.unsqueeze(-2),
                        t2.unsqueeze(-2),
                        t3.unsqueeze(-2)), dim=-2)   

        log_det = torch.sum(s0, dim=-1) + torch.sum(s1, dim=-1) + torch.sum(s2, dim=-1) + torch.sum(s3, dim=-1)
        log_det = torch.mean(log_det)       

        return z, ss, ts, log_det, flag
    
    
    # def forward(self, eps):
    #     z = torch.zeros_like(eps).to(device)
    #     L_slopes = torch.tensor([]).to(device)
    #     ss = torch.tensor([]).to(device)
    #     ts = torch.tensor([]).to(device)

    #     total_dims = self.L*self.ldim
    #     for i in range(self.L):
    #         if 1 in self.A[i, :]:
    #             mask = self.A[i, :].repeat(self.ldim, 1).T.reshape(total_dims).to(device)  
    #             # logger.info(f"computing child params...")
    #         else:
    #             mask = torch.zeros(total_dims).to(device)
    #             # logger.info(f"computing root params...")
            
    #         z[:, :, i, :], s, t = self.nf[str(i)](eps[:, :, i, :], z, mask)
    #         L_slopes = torch.cat((L_slopes, s.unsqueeze(-2)), dim=-2)
            
    #         ss = torch.cat((ss, s.unsqueeze(-2)), dim=-2)
    #         ts = torch.cat((ts, t.unsqueeze(-2)), dim=-2)

    #     return z, L_slopes, ss, ts


class PriorMultivariateCausalFlow(nn.Module):
    def __init__(self, ldim):
        super().__init__()
        
        self.s_base = nn.Parameter(torch.randn(1, 1, ldim), requires_grad=True)
        self.t_base = nn.Parameter(torch.randn(1, 1, ldim), requires_grad=True)
        self.s_cond = MLP(ldim, ldim, ldim*4)
        self.t_cond = MLP(ldim, ldim, ldim*4)

        self.s2_base = nn.Parameter(torch.randn(1, 1, ldim), requires_grad=True)
        self.t2_base = nn.Parameter(torch.randn(1, 1, ldim), requires_grad=True)

        self.s3_base = nn.Parameter(torch.randn(1, 1, ldim), requires_grad=True)
        self.t3_base = nn.Parameter(torch.randn(1, 1, ldim), requires_grad=True)

        self.layer_0 = nn.Linear(1, ldim)
        self.layer_1 = nn.Linear(1, ldim)
        self.layer_2 = nn.Linear(1, ldim)
        self.layer_3 = nn.Linear(1, ldim)

    def forward(self, u, z):
        B, N, _, _ = z.shape
        
        s0 = self.s_base.expand(B, N, -1)
        t0 = self.t_base.expand(B, N, -1)
        cp0 = torch.exp(s0)*self.layer_0(u[:, :, 0].unsqueeze(-1)) + t0

        # s1 = self.s_cond(z0)
        # t1 = self.t_cond(z0)
        s1 = self.s_cond(z[:, :, 0])
        t1 = self.t_cond(z[:, :, 0])
        cp1 = torch.exp(s1)*self.layer_1(u[:, :, 1].unsqueeze(-1)) + t1

        s2 = self.s2_base.expand(B, N, -1)
        t2 = self.t2_base.expand(B, N, -1)
        cp2 = torch.exp(s2)*self.layer_2(u[:, :, 2].unsqueeze(-1)) + t2

        s3 = self.s3_base.expand(B, N, -1)
        t3 = self.t3_base.expand(B, N, -1)
        cp3 = torch.exp(s3)*self.layer_3(u[:, :, 3].unsqueeze(-1)) + t3

        cp = torch.cat((cp0.unsqueeze(-2), cp1.unsqueeze(-2), cp2.unsqueeze(-2), cp3.unsqueeze(-2)), dim=-2)
        ss = torch.cat((s0.unsqueeze(-2), s1.unsqueeze(-2), s2.unsqueeze(-2), s3.unsqueeze(-2)), dim=-2)
        ts = torch.cat((t0.unsqueeze(-2), t1.unsqueeze(-2), t2.unsqueeze(-2), t3.unsqueeze(-2)), dim=-2)   

        log_det = torch.sum(s0, dim=-1) + torch.sum(s1, dim=-1) + torch.sum(s2, dim=-1) + torch.sum(s3, dim=-1)
        log_det = torch.mean(log_det)       

        return cp

class GaussianMLP(nn.Module):
    def __init__(self, nlayers, hid_act, nin, nout, nh, learn_var, alpha, beta):
        super().__init__()
        self.learn_var = learn_var

        self.mean = nn.Linear(nin, nout)
        if learn_var: self.logvar = nn.Linear(nin, nout)

        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        mean = self.alpha * self.mean(x)
        if self.learn_var: logvar = self.beta * self.logvar(x)
        else: logvar = torch.zeros_like(mean)
        return mean, logvar
        
class ConditionalGaussianPrior(nn.Module):
    def __init__(self, L, ldim, learn_var=False, clamp_var=False):
        super().__init__()
        self.learn_var = learn_var
        self.clamp_var = clamp_var
        self.L = L
        self.net = nn.ModuleDict()
        
        alpha = [0.2, 1., 1.5, 0.7] 
        beta = [1., 0.2, 0.7, 1.5]
        for i in range(self.L):
            self.net[str(i)] = GaussianMLP(1, nn.ELU(), 1, ldim, ldim, learn_var, alpha[i], beta[i])

    def forward(self, labels):
        means, logvars = [], []
        for i in range(self.L):
            out = self.net[str(i)](labels[:, :, i].unsqueeze(-1))
            assert isinstance(out, tuple)

            mean, logvar = out[0], out[1]
            if self.clamp_var:
                logvar = torch.clamp(logvar, -3., 2.)

            means.append(mean)   
            logvars.append(logvar)       
        means = torch.stack(means, -2)
        logvars = torch.stack(logvars, -2)
        return means, logvars

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
