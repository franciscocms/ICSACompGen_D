import torch
import torch.nn as nn

from contextlib import nullcontext
from typing import Any, Dict

import sys
sys.path.insert(0, '../..') # add parent-parent dir

from src.main import params

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
        self.layers = nn.Sequential(
            nn.Linear(nin, nhidden),
            nn.ELU(),
            nn.Linear(nhidden, nhidden),
            nn.ELU(),
            nn.Linear(nhidden, nhidden),
            nn.ELU(),
            nn.Linear(nhidden, nhidden),
            nn.ELU(),
            nn.Linear(nhidden, nout),
        )

    def forward(self, z):
        # z: [B, N, L*d]
        batch_size, n_slots, features = z.size()
        z = z.reshape(batch_size*n_slots, -1)
        out = self.layers(z)
        out = out.reshape(batch_size, n_slots, -1)
        return out

class NoiseEncoder(nn.Module):
    def __init__(self, nin, nh, nout, slot_dim):
        super().__init__()
        self.encoder = Encoder(nin, nh, nout, slot_dim) 
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

class MLP(nn.Module):
    def __init__(self, nin, nout, nh, f_act="identity"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            # nn.ELU(),
            # nn.Linear(nh, nh),
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
        
        nhidden = nin*4
        self.net = nn.Sequential(
            nn.Linear(nin, nhidden),
            nn.ELU(),
            nn.Linear(nhidden, nhidden),
            nn.ELU(),
            nn.Linear(nhidden, nout)
            #nn.Sigmoid()
        )
        #self.net = nn.Linear(nin, nout)

    def forward(self, slots):
        B, N, d = slots.shape
        out = self.net(slots)
        return out

class LinearProbe(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        
        # nhidden = nin*4
        # self.net = nn.Sequential(
        #     nn.Linear(nin, nhidden),
        #     nn.ELU(),
        #     nn.Linear(nhidden, nhidden),
        #     nn.ELU(),
        #     nn.Linear(nhidden, nout)
        # )
        self.net = nn.Linear(nin, nout)

    def forward(self, x):
        out = self.net(x)
        if out.size(-1) == 1: out = out.squeeze(-1)
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

        #s = s - s.mean(dim=-1, keepdim=True)
        
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
                # logger.info(f"computing child params...")
            else:
                mask = torch.zeros(total_dims).to(device)
                # logger.info(f"computing root params...")
            
            z[:, :, i, :], s, t = self.nf[str(i)](eps[:, :, i, :], z, mask)
            L_slopes = torch.cat((L_slopes, s.unsqueeze(-2)), dim=-2)
            
            ss = torch.cat((ss, s.unsqueeze(-2)), dim=-2)
            ts = torch.cat((ts, t.unsqueeze(-2)), dim=-2)

        return z, L_slopes, ss, ts

class GaussianMLP(nn.Module):
    def __init__(self, nlayers, hid_act, nin, nout, nh, learn_var):
        super().__init__()
        self.learn_var = learn_var
        # self.net = []
        # for _ in range(nlayers):
        #     self.net.append(nn.Linear(nin, nh))
        #     self.net.append(hid_act)
        #     nin = nh
        # self.net = nn.Sequential(*self.net)

        self.mean = nn.Linear(nin, nout)
        
        #if learn_var: self.logvar =  nn.Sequential(nn.Linear(nin, nout), nn.Sigmoid())
        if learn_var: self.logvar = nn.Linear(nin, nout)

    def forward(self, x):
        #x = self.net(x)
        if self.learn_var:
            return self.mean(x), self.logvar(x)
        else:
            return self.mean(x)    
        

class ConditionalGaussianPrior(nn.Module):
    def __init__(self, L, ldim, learn_var=False):
        super().__init__()
        self.learn_var = learn_var
        self.L = L
        self.net = nn.ModuleDict()
        for i in range(self.L):
            self.net[str(i)] = GaussianMLP(1, nn.ELU(), 1, ldim, ldim, learn_var)

    def forward(self, labels):
        means, logvars = [], []
        for i in range(self.L):
            out = self.net[str(i)](labels[:, :, i].unsqueeze(-1))
            if isinstance(out, tuple):
                means.append(out[0])   
                logvars.append(out[1]) 
            else:
                means.append(out)       
        means = torch.stack(means, -2)
        if len(logvars) > 0:
            logvars = torch.stack(logvars, -2)
            return means, logvars
        else:
            return means

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


######################################################################################################################################################

import torch.nn.functional as F
import numpy as np

from src.utils.guide import sample_z_from_latents

class SlotAttentionAutoEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        num_slots,
        num_iterations,
        hid_dim,
        softmax=True,
        sampling=True,
    ):
        """
        Builds the Slot Attention-based auto-encoder.

        Args:
            num_slots: Number of slots in Slot Attention.
            num_iterations: Number of iterations in Slot Attention.
            hid_dim: Hidden dimension of Slot Attention.
        """
        super().__init__()
        self.model_name = "SlotAttention"
        self.hid_dim = hid_dim
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.encoder_cnn = encoder
        self.decoder_cnn = decoder

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=hid_dim,
            iters=self.num_iterations,
            eps=1e-8,
            hidden_dim=64,
            use_sampling=sampling,
        )

        self.spatial_broadcast = 8

        self.use_softmax = softmax

    def encode(self, x):
        # `x` is an image which has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(x)  # CNN Backbone.
        x = F.layer_norm(x, x.shape[1:])
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        logger.info(f"5: {x.shape}")

        # Slot Attention module.
        return self.slot_attention(x)

    def decode(self, hat_z):
        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        slots = (
            hat_z.reshape((-1, hat_z.shape[-1]))
            .unsqueeze(1)
            .unsqueeze(2)
            .repeat((1, self.spatial_broadcast, self.spatial_broadcast, 1))
        )

        # logger.info(f"7 (slots broadcasted): {slots.shape}")

        # `out` has shape: [batch_size*num_slots, width, height, num_channels+1].
        out = self.decoder_cnn(slots)
        
        # logger.info(f"8 (decoder out): {out.shape}")

        # Undo combination of slot and batch dimension; split alpha masks.
        reconstructions, masks = out.reshape(
            -1, self.num_slots, out.shape[1], out.shape[2], out.shape[3]
        ).split([3, 1], dim=-1)

        # Normalize alpha masks over slots.
        if self.use_softmax:
            masks = F.softmax(masks, dim=1)
        else:
            masks = torch.sigmoid(masks)
        xhs = reconstructions * masks

        # `hat_x` has shape: [batch_size, num_channels, width, height].
        hat_x = torch.sum(xhs, dim=1).permute(0, 3, 1, 2)

        return hat_x, xhs, masks, reconstructions

    def forward(
        self,
        x,
        use_consistency_loss=False,
        extended_consistency_loss=False,
    ) -> Dict[str, Any]:
        hat_z = self.encode(x)
        
        logger.info(f"6 (slots): {hat_z.shape}")

        hat_x, figures, masks, reconstructions = self.decode(hat_z)

        output_dict = {
            "reconstructed_image": hat_x,
            "predicted_latents": hat_z,
            "reconstructed_figures": figures.permute(1, 0, 4, 2, 3),
            "reconstructed_masks": masks.permute(1, 0, 4, 2, 3),
            "raw_figures": reconstructions.permute(1, 0, 4, 2, 3),
        }
        return output_dict


class SlotAttention(nn.Module):
    def __init__(
        self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128, use_sampling=True
    ):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim**-0.5
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.use_sampling = use_sampling

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_log_sigma.expand(b, n_s, -1).exp()

        # remove randomness from slots initialization
        if not self.use_sampling:
            g = torch.Generator()
            g.manual_seed(0)

            slots = mu + sigma * torch.normal(0, 1, size=sigma.shape, generator=g).to(
                sigma.device
            )
        else:
            slots = mu + sigma * torch.randn_like(sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bjd,bij->bid", v, attn)

            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.elu(self.fc1(self.norm_pre_ff(slots))))

        return slots


def build_grid(resolution):
    ranges = [np.linspace(0.0, 1.0, num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))


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
        grid = self.embedding(self.grid.to(inputs.device))
        return inputs + grid


class SlotAttentionEncoder(nn.Module):
    def __init__(self, resolution, hid_dim, ch_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, ch_dim, 4, 2, 1)
        self.conv2 = nn.Conv2d(ch_dim, ch_dim, 4, 2, 1)
        self.conv3 = nn.Conv2d(ch_dim, ch_dim, 4, 2, 1)
        self.conv4 = nn.Conv2d(ch_dim, hid_dim, 4, 2, 1)

        resolution = (8, 8)
        self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)

    def forward(self, x):
        x = self.conv1(x)
        x = F.elu(x)
        x = self.conv2(x)
        x = F.elu(x)
        x = self.conv3(x)
        x = F.elu(x)
        x = self.conv4(x)
        x = F.elu(x)
        logger.info(f"1: {x.shape}")
        x = x.permute(0, 2, 3, 1)
        logger.info(f"2: {x.shape}")
        x = self.encoder_pos(x)
        logger.info(f"3: {x.shape}")
        x = torch.flatten(x, 1, 2)
        logger.info(f"4: {x.shape}")
        return x


class SlotAttentionDecoder(nn.Module):
    def __init__(self, ch_dim, hid_dim, resolution):
        super().__init__()

        self.decoder_initial_size = (8, 8)

        self.conv_list = nn.ModuleList(
            [nn.ConvTranspose2d(hid_dim, ch_dim, 4, 2, 1), nn.ELU()]
        )
        for i in range(2):
            self.conv_list.append(nn.ConvTranspose2d(ch_dim, ch_dim, 4, 2, 1))
            self.conv_list.append(nn.ELU())

        self.conv_list.append(nn.ConvTranspose2d(ch_dim, 4, 4, 2, 1))

        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        x = self.decoder_pos(x)
        logger.info(x.shape)
        x = x.permute(0, 3, 1, 2)
        for conv_layer in self.conv_list:
            x = conv_layer(x)
            logger.info(x.shape)

        x = x[:, :, : self.resolution[0], : self.resolution[1]]
        x = x.permute(0, 2, 3, 1)
        return x


