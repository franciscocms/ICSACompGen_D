import torch
import torch.nn as nn

from contextlib import nullcontext
from typing import Any, Dict

import sys
sys.path.insert(0, '../..') # add parent-parent dir

from src.main import params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



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
        return tensor.view(self.size)

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
        View((-1, 32 * 8 * 8)),
        nn.Linear(32 * 8 * 8, 256),
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
        nn.Linear(256, 32 * 8 * 8),
        nn.ELU(),
        View((-1, 32, 8, 8)),
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
        self.model_name = "SlotMLPAdditive"        

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
    def __init__(self, nin, nh, nout, slot_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(nin, nh),
            nn.ELU(),
            nn.Linear(nh, nh),
            nn.ELU()
        )
        self.means = nn.Linear(nh, nout)
        self.logvars = nn.Linear(nh, nout)
        #self.layer_norm = nn.LayerNorm(slot_dim)
    
    def forward(self, x):
        #x = self.layer_norm(x)
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

class NoiseEncoder(nn.Module):
    def __init__(self, nin, nh, nout, slot_dim):
        super().__init__()
        self.encoder = Encoder(nin, nh, nout, slot_dim) 

    def encode(self, x):
        mu, logvars = self.encoder(x)
        
        assert not torch.isnan(mu).any(), f"{mu}"
        assert not torch.isnan(logvars).any(), f"{logvars}"
        
        stds = torch.exp(0.5*logvars)
        z = mu + torch.randn_like(mu)*stds # reparameterization trick
        return mu, logvars, z
    
class CausalDecoder(nn.Module):
    def __init__(self, nin, nh, nout):  
        super().__init__()
        self.decoder = Decoder(nin=nin, nh=nh, nout=nout)    # takes the reparameterized z to generate hat(u_m)
        n_latents, latent_dim = len(params["latent_vars"]), params["latent_dim"]
        #self.layer_norm = nn.LayerNorm(n_latents*latent_dim)

    def forward(self, z):
        #z = self.layer_norm(z)
        rec = self.decoder(z)
        return rec

class MLP(nn.Module):
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

        # nn.init.zeros_(self.net[-1].weight)
        # nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, mask=None):
        if mask == None: 
            return self.f_act(self.net(x))
        else:
            return self.f_act(self.net(x * mask))
        

class Predictor(nn.Module):
    def __init__(self, nin, nhidden, nout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nhidden),
            nn.ELU(),
            nn.Linear(nhidden, nhidden),
            nn.ELU(),
            nn.Linear(nhidden, nout),
            nn.Sigmoid()
        )

    def forward(self, slots):
        B, N, d = slots.shape
        out = self.net(slots)
        return out

class LinearProbe(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nin),
            nn.ReLU(),
            nn.Linear(nin, nout),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.net(x)
        if out.size(-1) == 1: out = out.squeeze(-1)
        return out

class Flow(nn.Module):
    def __init__(self, L, ldim, A=None, net_class=MLP, nh=100, f_act="identity"):
        super().__init__()

        self.s_cond = net_class(L*ldim, ldim, nh, f_act=f_act)
        self.t_cond = net_class(L*ldim, ldim, nh, f_act=f_act)
        self.total_dims = L*ldim
        
        self.L, self.ldim = L, ldim

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
            self.nf[str(i)] = Flow(L, ldim, A, f_act="identity") # instead of "tanh"

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


class FlexibleGaussianMLP(nn.Module):
    def __init__(self, nlayers, hid_act, nin, nout, nh):
        super().__init__()
        self.net = []
        for _ in range(nlayers-1):
            self.net.append(nn.Linear(nin, nh))
            self.net.append(hid_act)
            nin = nh
        self.net = nn.Sequential(*self.net)

        self.mean = nn.Linear(nh, nout)
        self.logvar = nn.Linear(nh, nout)

    def forward(self, x):
        x = self.net(x)
        return self.mean(x), self.logvar(x)


class ConditionalGaussianPrior(nn.Module):
    def __init__(self, L, ldim):
        super().__init__()

        self.L = L
        self.net = nn.ModuleDict()
        for i in range(self.L):
            self.net[str(i)] = FlexibleGaussianMLP(2, nn.ELU(), ldim, ldim, ldim)

    def forward(self, labels):
        means, logvars = [], []
        for i in range(self.L):
            out = self.net[str(i)](labels[:, :, i, :])
            means.append(out[0])
            logvars.append(out[1])            
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
