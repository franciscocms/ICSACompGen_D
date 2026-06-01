import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from contextlib import nullcontext
from typing import Any, Dict

import sys
sys.path.insert(0, '../..') # add parent-parent dir

from srcc.main import params
from srcc.utils.guide import permute_dims, inverse_permutation

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
    if out_channels == 1:
        module_list.append(nn.Sigmoid())

    decoder = nn.Sequential(*module_list)
    return decoder



class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, geo_dim):
        super().__init__()
        # Generates scale and shift for every single neuron in this layer
        self.conditioner = nn.Linear(geo_dim, 2 * feature_dim)
        
    def forward(self, x, z_geo):
        params = self.conditioner(z_geo)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        # Softplus ensures gamma > 0, keeping the cross-derivative non-zero
        return F.softplus(gamma) * x + beta

class CompositionalObjectDecoder(nn.Module):
    def __init__(self, ldim, hidden_dim=256, img_size=64):
        super().__init__()
        self.img_size = img_size
        
        self.app_dim = ldim 
        self.geo_dim = ldim 

        # --- The Appearance Trunk ---
        # This path only 'sees' Color. It learns a generic 'object-ness'.
        self.fc1 = nn.Linear(self.app_dim, hidden_dim)
        self.film1 = FiLMLayer(hidden_dim, self.geo_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.film2 = FiLMLayer(hidden_dim, self.geo_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.film3 = FiLMLayer(hidden_dim, self.geo_dim)

        self.to_pixels = nn.Sequential(
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
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, z):
        z0, z1, z2, z3 = torch.split(z, params["latent_dim"], -1)

        # Layer 1: Start with Appearance, modulate with Geometry
        x = self.fc1(z0)
        x = self.film1(x, z1)
        x = F.elu(x)

        # Layer 2: Deepen the interaction
        x = self.fc2(x)
        x = self.film2(x, z2)
        x = F.elu(x)

        # Layer 3: Final refinement
        x = self.fc3(x)
        x = self.film3(x, z3)
        x = F.elu(x)

        # Output to image
        out = self.to_pixels(x)
        return out.view(-1, 3, self.img_size, self.img_size)
    
    def old_forward(self, z):
        # z shape: (B, L * ldim)
        # Manually slice the composition (assuming fixed order from your prior)
        # This is where the causal structure is enforced
        z_app = z[:, self.app_dim:2*self.app_dim]
        z_geo = torch.cat((z[:, :self.app_dim],
                           z[:, 2*self.app_dim:]), dim=-1)

        # Layer 1: Start with Appearance, modulate with Geometry
        x = self.fc1(z_app)
        x = self.film1(x, z_geo)
        x = F.elu(x)

        # Layer 2: Deepen the interaction
        x = self.fc2(x)
        x = self.film2(x, z_geo)
        x = F.elu(x)

        # Layer 3: Final refinement
        x = self.fc3(x)
        x = self.film3(x, z_geo)
        x = F.elu(x)

        # Output to image
        out = self.to_pixels(x)
        return out.view(-1, 3, self.img_size, self.img_size)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()

        self.net = nn.Sequential(
            # B x 4
            nn.Linear(latent_dim, 256),
            nn.ELU(),

            nn.Linear(256, 512),
            nn.ELU(),

            nn.Linear(512, 1024),
            nn.ELU(),

            # Project into feature map
            nn.Linear(1024, 256 * 8 * 8),
            nn.ELU()
        )

        self.conv_net = nn.Sequential(
            # B x 256 x 8 x 8

            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ELU(),

            # B x 128 x 16 x 16

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ELU(),

            # B x 64 x 32 x 32

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ELU(),

            # B x 32 x 64 x 64

            nn.Conv2d(
                32, 3,
                kernel_size=3,
                padding=1
            ),

            nn.Sigmoid()
        )

    def forward(self, z):
        """
        z: Tensor of shape (B, 4)
        """

        x = self.net(z)

        # reshape into spatial feature map
        x = x.view(-1, 256, 8, 8)

        x = self.conv_net(x)

        return x

class SlotMLPAdditiveDecoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_slots: int,
        n_slot_latents: int,
    ) -> None:
        super(SlotMLPAdditiveDecoder, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        
        self.z1_decoder = ConvDecoder()
        self.z2_decoder = ConvDecoder()
        self.z3_decoder = ConvDecoder()
        self.z4_decoder = ConvDecoder()
        
        self.model_name = "SlotMLPAdditiveDecoder"

    def build_decoder(self, in_channels, out_channels):
        module_list = [
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)       
        ]
        if out_channels == 1:
            module_list.append(nn.Sigmoid())
        decoder = nn.Sequential(*module_list)
        return decoder

    def sbd(self, zi, B, N):
        zi = zi.view(zi.squeeze().shape + (1, 1))
        zi = zi.expand((-1, -1, params["resolution"][0], params["resolution"][1]))
        x = torch.linspace(-1, 1, params["resolution"][0]).to(device)
        y = torch.linspace(-1, 1, params["resolution"][1]).to(device)
        x_grid, y_grid = torch.meshgrid(x, y)
        x_grid = x_grid.view((1, 1) + x_grid.shape)
        y_grid = y_grid.view((1, 1) + y_grid.shape)
        zi_grid = torch.cat((x_grid.expand(B*N, -1, -1, -1),
                            y_grid.expand(B*N, -1, -1, -1), zi), dim=1)
        return zi_grid

    def forward(self, latents):
        # Reshape latents from [batch_size, n_slots, features] to [batch_size*n_slots, features]
        # batch_size, n_slots, features = latents.size()
        # reshaped_latents = latents.view(batch_size * n_slots, features)
        batch_size, n_slots, n_features, features = latents.size()
        reshaped_latents = latents.view(batch_size * n_slots, n_features, features)

        H, W = params["resolution"]
        
        z1, z2, z3, z4 = torch.split(reshaped_latents, 1, 1)
        # z1 = self.sbd(z1, batch_size, n_slots)
        # z2 = self.sbd(z2, batch_size, n_slots)
        # z3 = self.sbd(z3, batch_size, n_slots)
        # z4 = self.sbd(z4, batch_size, n_slots)

        # Pass reshaped latents through decoder
        reshaped_figures = self.z1_decoder(z1.squeeze()) * self.z2_decoder(z2.squeeze()) * self.z3_decoder(z3.squeeze()) * self.z4_decoder(z4.squeeze())
        #logger.info(reshaped_figures.shape)

        # Reshape figures back to [batch_size, n_slots, ...]
        figures = reshaped_figures.view(
            batch_size, n_slots, *reshaped_figures.shape[1:]
        )
        # Sum over the n_slots dimension for image  
        image = figures.sum(dim=1)

        return image, figures

"""
uncomment if to use the multiplicative decoder with SBD inspiration for each f_app and f_geom
f(z) = f_app(z_color) + f_geom(z_geom), where z_geom = cat(z_scale, z_x, z_y)
"""

# class SlotMLPAdditiveDecoder(torch.nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         n_slots: int,
#         n_slot_latents: int,
#     ) -> None:
#         super(SlotMLPAdditiveDecoder, self).__init__()
#         self.n_slots = n_slots
#         self.n_slot_latents = n_slot_latents
        
#         self.app_decoder = nn.Sequential(
#             nn.Conv2d(6, 64, kernel_size=3, padding=1),
#             nn.ELU(),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ELU(),
#             nn.Conv2d(64, 3, kernel_size=3, padding=1)
#         )
        
#         self.geom_decoder = nn.Sequential(
#             nn.Conv2d(14, 64, kernel_size=3, padding=1),
#             nn.ELU(),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ELU(),
#             nn.Conv2d(64, 1, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
#         self.model_name = "SlotMLPAdditiveDecoder"

#     def forward(self, latents):
#         # Reshape latents from [batch_size, n_slots, features] to [batch_size*n_slots, features]
#         # batch_size, n_slots, features = latents.size()
#         # reshaped_latents = latents.view(batch_size * n_slots, features)
#         batch_size, n_slots, n_features, features = latents.size()
#         reshaped_latents = latents.view(batch_size * n_slots, n_features, features)

#         H, W = params["resolution"]
        
#         z_app = reshaped_latents[:, 1]
#         z_geom = torch.flatten(reshaped_latents[:, [0, 2, 3]], -2)

#         z_app = z_app.view(z_app.shape + (1, 1))
#         z_geom = z_geom.view(z_geom.shape + (1, 1))

#         z_app = z_app.expand((-1, -1, H, W))
#         z_geom = z_geom.expand((-1, -1, H, W))

#         x = torch.linspace(-1, 1, W).to(device)
#         y = torch.linspace(-1, 1, H).to(device)
#         x_grid, y_grid = torch.meshgrid(x, y)
#         x_grid = x_grid.view((1, 1) + x_grid.shape)
#         y_grid = y_grid.view((1, 1) + y_grid.shape)

#         z_app_grid = torch.cat((x_grid.expand(batch_size * n_slots, -1, -1, -1),
#                                 y_grid.expand(batch_size * n_slots, -1, -1, -1), z_app), dim=1)

#         z_geom_grid = torch.cat((x_grid.expand(batch_size * n_slots, -1, -1, -1),
#                                 y_grid.expand(batch_size * n_slots, -1, -1, -1), z_geom), dim=1)

#         # Pass reshaped latents through decoder
#         reshaped_figures = self.app_decoder(z_app_grid) * self.geom_decoder(z_geom_grid)
#         #logger.info(reshaped_figures.shape)

#         # Reshape figures back to [batch_size, n_slots, ...]
#         figures = reshaped_figures.view(
#             batch_size, n_slots, *reshaped_figures.shape[1:]
#         )
#         # Sum over the n_slots dimension for image  
#         image = figures.sum(dim=1)

#         return image, figures


"""
uncomment if to use the standard multiplicative decoder
f(z) = f_app(z_color) + f_geom(z_geom), where z_geom = cat(z_scale, z_x, z_y)
"""

# class SlotMLPAdditiveDecoder(torch.nn.Module):
#     """
#     Model generates x_hat = sum_{i=1}^{n_slots} f(z_i),
#     by summing the output of each slot.  Model outputs x_hat and list of x_hat_i.
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         n_slots: int,
#         n_slot_latents: int,
#     ) -> None:
#         super(SlotMLPAdditiveDecoder, self).__init__()
#         self.n_slots = n_slots
#         self.n_slot_latents = n_slot_latents
#         self.app_decoder = get_decoder(params["latent_dim"], 3)
#         self.geom_decoder = get_decoder(params["latent_dim"]*3, 1)
#         self.model_name = "SlotMLPAdditiveDecoder"

#     def forward(self, latents):
#         # Reshape latents from [batch_size, n_slots, features] to [batch_size*n_slots, features]
#         # batch_size, n_slots, features = latents.size()
#         # reshaped_latents = latents.view(batch_size * n_slots, features)
#         batch_size, n_slots, n_features, features = latents.size()
#         reshaped_latents = latents.view(batch_size * n_slots, n_features, features)

#         # Pass reshaped latents through decoder
#         reshaped_figures = self.app_decoder(reshaped_latents[:, 1]) * self.geom_decoder(torch.flatten(reshaped_latents[:, [0, 2, 3]], -2))
#         #logger.info(reshaped_figures.shape)

#         # Reshape figures back to [batch_size, n_slots, ...]
#         figures = reshaped_figures.view(
#             batch_size, n_slots, *reshaped_figures.shape[1:]
#         )
#         # Sum over the n_slots dimension for image  
#         image = figures.sum(dim=1)

#         return image, figures

class old_SlotMLPAdditiveDecoder(torch.nn.Module):
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
        super(old_SlotMLPAdditiveDecoder, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.decoder = get_decoder(n_slot_latents, in_channels)
        #self.decoder = CompositionalObjectDecoder(params["latent_dim"])
        self.model_name = "SlotMLPAdditiveDecoder"

    def forward(self, latents):
        # Reshape latents from [batch_size, n_slots, features] to [batch_size*n_slots, features]
        # batch_size, n_slots, features = latents.size()
        # reshaped_latents = latents.view(batch_size * n_slots, features)
        batch_size, n_slots, n_features, features = latents.size()
        reshaped_latents = latents.view(batch_size * n_slots, n_features, features)

        # Pass reshaped latents through decoder
        reshaped_figures = self.decoder(reshaped_latents)
        #logger.info(reshaped_figures.shape)

        # Reshape figures back to [batch_size, n_slots, ...]
        figures = reshaped_figures.view(
            batch_size, n_slots, n_features, *reshaped_figures.shape[1:]
        ).sum(dim=2)
        # Sum over the n_slots dimension for image  
        image = figures.sum(dim=1)

        # 3. Split into RGB and Mask Logits
        # figures_rgb = figures[:, :, :3, :, :]
        # figures_mask = figures[:, :, 3:, :, :]            
        
        # # 4. ENFORCE COMPOSITIONALITY: Softmax over the Slot Dimension (N)
        # # This ensures that for every pixel, the masks sum to 1.0 across all slots.
        # figures_mask = torch.nn.functional.softmax(figures_mask, dim=1)             # [B, N, 1, H, W]
        
        # # 5. Composite the final image
        # # This is the "Additive" step constrained by the disjoint masks
        # image = torch.sum(figures_rgb * figures_mask, dim=1) # [B, 3, H, W]

        return image, figures
    
class FullNoiseEncoder(nn.Module):
    def __init__(self, in_channels, n_slots, n_latents, latent_dim, stage="train", clamp_var=False):
        super().__init__()

        self.stage = stage
        self.clamp_var = clamp_var
        out_dim = n_slots * n_latents * latent_dim
        # out_dim = n_slots * latent_dim
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
        # mean = self.fc_mean(x).view(-1, self.n_slots, self.latent_dim)
        # logvar = self.fc_logvar(x).view(-1, self.n_slots, self.latent_dim)
        if self.clamp_var: logvar = torch.clamp(logvar, -3., 3.)
        return mean, logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        stds = torch.exp(0.5*logvar)
        z = mu + torch.randn_like(mu)*stds
        return mu, logvar, z
    

class AddCoords(nn.Module):
    """ Adds (x, y) coordinates as extra channels to help measure scale/position. """
    def forward(self, x):
        batch_size, _, h, w = x.size()
        xx_range = torch.arange(w, device=x.device).float() / (w - 1) * 2 - 1
        yy_range = torch.arange(h, device=x.device).float() / (h - 1) * 2 - 1
        # Create 2D grids
        xx_channel = xx_range.view(1, 1, 1, w).expand(batch_size, 1, h, w)
        yy_channel = yy_range.view(1, 1, h, 1).expand(batch_size, 1, h, w)
        return torch.cat([x, xx_channel, yy_channel], dim=1)

class ScaleSensitiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        # Use 3x3 for better feature extraction before the 'jump'
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # Use stride in a separate layer to decouple feature learning from downsampling
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1)
        self.norm = nn.GroupNorm(8, out_channels) # Better than BatchNorm for small batches
        self.act = nn.ELU() # ELU avoids the "flat" bins of ReLU

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.norm(self.conv2(x)))
        return x

class RobustEncoder(nn.Module):
    def __init__(self, n_slots, n_latents, latent_dim, input_channels=3):
        super().__init__()
        self.add_coords = AddCoords()
        
        out_dim = n_slots * n_latents * latent_dim
        # Initial layer sees 3 RGB + 2 Coord channels = 5
        self.layer1 = ScaleSensitiveBlock(input_channels + 2, 32) # 64 -> 32
        self.layer2 = ScaleSensitiveBlock(32, 64)                # 32 -> 16
        self.layer3 = ScaleSensitiveBlock(64, 128)               # 16 -> 8
        self.layer4 = ScaleSensitiveBlock(128, 256)              # 8 -> 4
        
        # Avoid GlobalAveragePooling; use MaxPool to preserve 'edge' signals for scale
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(256, 128)

        self.fc_mean = nn.Linear(128, out_dim)
        self.fc_logvar = nn.Linear(128, out_dim)

        self.n_latents = n_latents
        self.latent_dim = latent_dim
        self.n_slots = n_slots

    def encode(self, x):
        x = self.add_coords(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).view(x.size(0), -1)
        x = F.elu(self.fc(x))

        mean = self.fc_mean(x).view(-1, self.n_slots, self.n_latents*self.latent_dim)
        logvar = self.fc_logvar(x).view(-1, self.n_slots, self.n_latents*self.latent_dim)
    
        return mean, logvar
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        stds = torch.exp(0.5*logvar)
        z = mu + torch.randn_like(mu)*stds
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

        # self.net = nn.Sequential(
        #     nn.Linear(nin, nh),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(nh, nh),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(nh, nh),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(nh, nout),
        # )

        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.Softplus(),
            nn.Linear(nh, nh),
            nn.Softplus(),
            nn.Linear(nh, nh),
            nn.Softplus(),
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
    def __init__(self, L, ldim, A, stage="train"):
        super().__init__()
        
        self.s_base = nn.Parameter(torch.randn(1, 1, ldim), requires_grad=True)
        self.t_base = nn.Parameter(torch.randn(1, 1, ldim), requires_grad=True)

        self.s2_base = nn.Parameter(torch.randn(1, 1, ldim), requires_grad=True)
        self.t2_base = nn.Parameter(torch.randn(1, 1, ldim), requires_grad=True)

        self.s3_base = nn.Parameter(torch.randn(1, 1, ldim), requires_grad=True)
        self.t3_base = nn.Parameter(torch.randn(1, 1, ldim), requires_grad=True)

        self.s_cond = MLP(ldim, ldim, ldim*4)
        self.t_cond = MLP(ldim, ldim, ldim*4)

        self.stage = stage
        self.scale_gain = 1.
        self.shift_gain = 1.
        

    def forward(self, eps, epoch=0, use_cc=False, debug=False):        
        
        B, N, _, _ = eps.shape

        s0 = self.s_base.expand(B, N, -1)
        t0 = self.t_base.expand(B, N, -1)
        z0 = torch.exp(s0)*eps[:, :, 0, :] + t0
          
        s1 = self.s_cond(z0) * self.scale_gain
        t1 = self.t_cond(z0) * self.shift_gain
        z1 = torch.exp(s1)*eps[:, :, 1, :] + t1

        s2 = self.s2_base.expand(B, N, -1)
        t2 = self.t2_base.expand(B, N, -1)
        z2 = torch.exp(s2)*eps[:, :, 2, :] + t2

        s3 = self.s3_base.expand(B, N, -1)
        t3 = self.t3_base.expand(B, N, -1)
        z3 = torch.exp(s3)*eps[:, :, 3, :] + t3

        if debug:
            logger.info(f"\ns0: {s0[0, 0]}")
            logger.info(f"t0: {t0[0, 0]}")
            logger.info(f"z0: {z0}")

            logger.info(f"\ns1: {s1}")
            logger.info(f"t1: {t1}")
            logger.info(f"z1: {z1}")

            logger.info(f"\ns2: {s2[0, 0]}")
            logger.info(f"t2: {t2[0, 0]}")
            logger.info(f"z2: {z2}")

            logger.info(f"\ns3: {s3[0, 0]}")
            logger.info(f"t3: {t3[0, 0]}")
            logger.info(f"z3: {z3}")

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

        return z, ss, ts, log_det

    
    def counterfactuals(self, eps, z, flag=None):
            
        B, N, _, _ = z.shape

        z0, z1, z2, z3 = torch.split(torch.flatten(z, -2), params["latent_dim"], -1)

        self.flag = torch.distributions.Categorical(
            torch.tensor([1/len(params["latent_vars"]) for _ in range(len(params["latent_vars"]))])
            ).sample() if flag == None else flag
        
        # z1 is [B, N, d]
        
        if self.flag == 0:
            z0_int, self.perm = permute_dims(torch.flatten(z0, 0, 1), z0.size(-1))
            z0_int = z0_int.view(B, N, -1)#.detach()
            
            s1_int = self.s_cond(z0_int) * self.scale_gain
            t1_int = self.t_cond(z0_int) * self.shift_gain
            z1_int = torch.exp(s1_int)*eps[:, :, 1, :] + t1_int

            self.z_int = torch.cat((z0_int.unsqueeze(-2),
                                    z1_int.unsqueeze(-2),
                                    z2.unsqueeze(-2),
                                    z3.unsqueeze(-2)), dim=-2)
            
            eps0_int = (z0_int - self.t_base.expand(B, N, -1)) * torch.exp(-self.s_base.expand(B, N, -1))
            self.eps_int = torch.cat((eps0_int.unsqueeze(-2), eps[:, :, 1:]), dim=-2)

        elif self.flag == 1:
            z1_perm, self.perm = permute_dims(torch.flatten(z1, 0, 1), z1.size(-1))
            z1_perm = z1_perm.view(B, N, -1)
            z0_perm = torch.flatten(z0, 0, 1)[self.perm]
            z0_perm = z0_perm.view(B, N, -1)
            s1_int = self.s_cond(z0_perm) * self.scale_gain
            t1_int = self.t_cond(z0_perm) * self.shift_gain
            e1_int = (z1_perm - t1_int) * torch.exp(-s1_int)
            
            #e1_int = e1_int.detach()
            z1_int = torch.exp(self.s_cond(z0) * self.scale_gain) * e1_int + self.t_cond(z0) * self.shift_gain

            self.z_int = torch.cat((z0.unsqueeze(-2),
                                    z1_int.unsqueeze(-2),
                                    z2.unsqueeze(-2),
                                    z3.unsqueeze(-2)), dim=-2)
            
            eps1_int = (z1_int - self.t_cond(z1)*self.shift_gain) * torch.exp(-self.s_cond(z1) * self.scale_gain)
            self.eps_int = torch.cat((eps[:, :, 0].unsqueeze(-2), eps1_int.unsqueeze(-2), eps[:, :, 2:]), dim=-2)

        elif self.flag == 2:
            z2_int, self.perm = permute_dims(torch.flatten(z2, 0, 1), z2.size(-1))
            z2_int = z2_int.view(B, N, -1)#.detach()
            self.z_int = torch.cat((z0.unsqueeze(-2),
                                    z1.unsqueeze(-2),
                                    z2_int.unsqueeze(-2),
                                    z3.unsqueeze(-2)), dim=-2)
            eps2_int = (z2_int - self.t2_base.expand(B, N, -1)) * torch.exp(-self.s2_base.expand(B, N, -1))
            self.eps_int = torch.cat((eps[:, :, :2], eps2_int.unsqueeze(-2), eps[:, :, -1].unsqueeze(-2)), dim=-2)
        
        elif self.flag == 3:
            z3_int, self.perm = permute_dims(torch.flatten(z3, 0, 1), z3.size(-1))
            z3_int = z3_int.view(B, N, -1)#.detach()
            self.z_int = torch.cat((z0.unsqueeze(-2),
                                    z1.unsqueeze(-2),
                                    z2.unsqueeze(-2),
                                    z3_int.unsqueeze(-2)), dim=-2)
            eps3_int = (z3_int - self.t3_base.expand(B, N, -1)) * torch.exp(-self.s3_base.expand(B, N, -1))
            self.eps_int = torch.cat((eps[:, :, :3], eps3_int.unsqueeze(-2)), dim=-2)
        
        return self.z_int, self.eps_int, self.flag
        
 
    def backward(self, z, flag=None, no_flow=False):

        B, N, _, d = z.shape
        
        if no_flow:
            ss = torch.zeros_like(z)
            ts = torch.zeros_like(z)
            log_det = 0.
            return z, ss, ts, log_det

        if flag == None:
            s0 = self.s_base.expand(B, N, -1)
            t0 = self.t_base.expand(B, N, -1)
            e0 = (z[:, :, 0] - t0) * torch.exp(-s0)

            s1 = self.s_cond(z[:, :, 0]) * self.scale_gain
            t1 = self.t_cond(z[:, :, 0]) * self.shift_gain
            e1 = (z[:, :, 1] - t1) * torch.exp(-s1)

            s2 = self.s_base.expand(B, N, -1)
            t2 = self.t_base.expand(B, N, -1)
            e2 = (z[:, :, 2] - t2) * torch.exp(-s2)

            s3 = self.s_base.expand(B, N, -1)
            t3 = self.t_base.expand(B, N, -1)
            e3 = (z[:, :, 3] - t3) * torch.exp(-s3)

            log_det = torch.sum(s0, dim=-1) + torch.sum(s1, dim=-1) + torch.sum(s2, dim=-1) + torch.sum(s3, dim=-1)
            log_det = torch.mean(log_det) 

            e = torch.cat((e0.unsqueeze(-2),
                           e1.unsqueeze(-2),
                           e2.unsqueeze(-2),
                           e3.unsqueeze(-2)), dim=-2)
            
            ss = torch.cat((s0.unsqueeze(-2),
                        s1.unsqueeze(-2),
                        s2.unsqueeze(-2),
                        s3.unsqueeze(-2)), dim=-2)
            ts = torch.cat((t0.unsqueeze(-2),
                        t1.unsqueeze(-2),
                        t2.unsqueeze(-2),
                        t3.unsqueeze(-2)), dim=-2) 
            
            return e, ss, ts, log_det
   
        elif flag == 0:
            
            # z = [z0_int, z1_int, z2, z3]

            s0 = self.s_base.expand(B, N, -1)
            t0 = self.t_base.expand(B, N, -1)
            e0_int = (z[:, :, 0] - t0) * torch.exp(-s0)
            
            # logger.info(z[:, :, 0])
            # logger.info(s0)
            # logger.info(t0)

            return e0_int

        elif flag == 1:
            # here we have to get the z_scale that took the influence on z_color 
            # because in this case, z_scale and z_color will have a different order

            # z = [z0, z1_int, z2, z3]
            
            z0_perm = torch.flatten(z[:, :, 0], 0, 1)[self.perm]
            assert z0_perm.shape == (B*N, d), f"{z0_perm.shape}"
            
            s1 = self.s_cond(z0_perm.view(B, N, -1)) * self.scale_gain
            t1 = self.t_cond(z0_perm.view(B, N, -1)) * self.shift_gain
            e1_int = (z[:, :, 1] - t1) * torch.exp(-s1)
            return e1_int    
    
        else: raise ValueError(f"invalid flag {flag}...")
    


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
        self.ss = torch.cat((s0.unsqueeze(-2), s1.unsqueeze(-2), s2.unsqueeze(-2), s3.unsqueeze(-2)), dim=-2)
        self.ts = torch.cat((t0.unsqueeze(-2), t1.unsqueeze(-2), t2.unsqueeze(-2), t3.unsqueeze(-2)), dim=-2)   
        self.s0, self.s1, self.s2, self.s3 = s0, s1, s2, s3
        self.log_det = torch.sum(s0, dim=-1) + torch.sum(s1, dim=-1) + torch.sum(s2, dim=-1) + torch.sum(s3, dim=-1)
        self.log_det = torch.mean(self.log_det)       
        return cp

class GaussianMLP(nn.Module):
    def __init__(self, nin, nout, nh, learn_var, alpha, beta):
        super().__init__()
        self.learn_var = learn_var
        # self.shared = nn.Sequential(
        #     nn.Linear(nin, nh),
        #     nn.ELU(),
        #     nn.Linear(nh, nh),
        #     nn.ELU(),
        #     nn.Linear(nh, nh),
        #     nn.ELU()
        # )
        
        self.mean_head = nn.Linear(nin, nout)
        if learn_var: self.logvar_head = nn.Linear(nin, nout)

        self.alpha = alpha
        self.beta = beta
        self.nout = nout

    def forward(self, x):
        #x = self.shared(x)
        mean = self.alpha * self.mean_head(x)
        if self.learn_var: logvar = self.beta * self.logvar_head(torch.square(x))
        else: logvar = torch.zeros_like(mean)
        return mean, logvar
        
class ConditionalGaussianPrior(nn.Module):
    def __init__(self, L, ldim, learn_var=False, clamp_var=False):
        super().__init__()
        self.learn_var = learn_var
        self.clamp_var = clamp_var
        self.L = L
        self.net = nn.ModuleDict()
        # alpha = [2., 3., 1.5, 0.7] 
        # beta = [3., 2., 4., 1.5]
        alpha = [1., 1., 1., 1.]
        beta = [1., 1., 1., 1.]
        self.freq_coeff = []
        for i in range(self.L):
            slope = 0.01 #+ (i * 0.05)
            freq = (i - 6) * np.pi
            self.freq_coeff.append(freq)
            self.net[str(i)] = GaussianMLP(1, 1, 32, learn_var, alpha[i], beta[i])
            #self.net[str(i)] = GaussianMLP(2, ldim, ldim*4, learn_var, alpha[i], beta[i])

    def forward(self, labels):

        B, N, _ = labels.shape
        means, logvars = [], []
        for i in range(self.L):
            #u_freq = torch.cat((labels[:, :, i].unsqueeze(-1), torch.sin(self.freq_coeff[i] * labels[:, :, i].unsqueeze(-1))), dim=-1)
            out = self.net[str(i)](labels[:, :, i].unsqueeze(-1))
            #out = self.net[str(i)](u_freq)
            
            assert isinstance(out, tuple)
            mean, logvar = out[0], out[1]
            if self.clamp_var:
                logvar = torch.clamp(logvar, -2., 2.)
            means.append(mean)   
            logvars.append(logvar)       
        means = torch.stack(means, -2)
        logvars = torch.stack(logvars, -2) - 3. #if params["jobID"] == 270 else torch.stack(logvars, -2) - 3.
        return means, logvars
    
class CausalGaussianPrior(nn.Module):
    def __init__(self, L, ldim, A):
        super().__init__()
        self.L = L
        self.A = A
        self.net = nn.ModuleDict()
        alpha = [1., 1., 1., 1.]
        beta = [1., 1., 1., 1.]
        self.freq_coeff = []
        for i in range(self.L):
            freq = (i - 6) * np.pi
            self.freq_coeff.append(freq)
            # self.net[str(i)] = GaussianMLP(nin=ldim + 2, nout=ldim, nh=ldim*4, learn_var=True,
            #                                 alpha=alpha[i], beta=beta[i])
            self.net[str(i)] = GaussianMLP(nin=ldim + 1, nout=ldim, nh=ldim*4, learn_var=True,
                                            alpha=alpha[i], beta=beta[i])

    def forward(self, labels, z):

        means, logvars = [], []
        for i in range(self.L):
            if 1 in self.A[i, :]:
                parent = (self.A[i, :] == 1).nonzero(as_tuple=False).item() # index of parent
                pa_zi = z[:, :, parent]
            else:
                pa_zi = torch.zeros_like(z[:, :, 0]).to(device)
            
            # zu_i = torch.cat((pa_zi,
            #                   labels[:, :, i].unsqueeze(-1), 
            #                   torch.sin(self.freq_coeff[i] * labels[:, :, i].unsqueeze(-1) * torch.norm(pa_zi, 2))), dim=-1)
            #zu_i = torch.cat((pa_zi, labels[:, :, i].unsqueeze(-1), torch.sin(self.freq_coeff[i] * labels[:, :, i].unsqueeze(-1))), dim=-1)
            zu_i = torch.cat((pa_zi, labels[:, :, i].unsqueeze(-1)), dim=-1)
            
            # logger.info(f"\nvar {i}")
            # logger.info(f"zu: {zu_i}")
            # for name, p in self.net[str(i)].named_parameters():
            #     logger.info(f"{p.T}")

            out = self.net[str(i)](zu_i)
            
            assert isinstance(out, tuple)
            mean, logvar = out[0], out[1]
            logvar = -torch.nn.functional.softplus(logvar) - 1.

            means.append(mean)   
            logvars.append(logvar)       
        means = torch.stack(means, -2)
        logvars = torch.stack(logvars, -2) #- 1. if params["jobID"] == 270 else torch.stack(logvars, -2) - 3.
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
