import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from contextlib import nullcontext
from typing import Any, Dict

import sys
sys.path.insert(0, '../..') # add parent-parent dir

from src.main import params
from src.utils.guide import minimize_entropy_of_sinkhorn, sinkhorn, assert_shape

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging
logger = logging.getLogger(params["running_type"])

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
        a = self.mlp_weight_input(inputs).squeeze(-1).softmax(-1) * n_s # 'a' shape (b_s, W*H)

        for _ in range(self.iters):            
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

            slots = self.gru(
                    updates.view(b_s * n_s, d),
                    slots_prev.view(b_s * n_s, self.slot_dim),
            )

            slots = slots.view(b_s, n_s, self.slot_dim)
            assert_shape(slots.size(), (b_s, n_s, self.slot_dim))
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
            assert_shape(slots.size(), (b_s, n_s, self.slot_dim))    
        return slots, attn

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
        self.encoder_sa = nn.Sequential(*self.encoder_sa)
        
        resolution = (32, 32)
        self.last_dim = 64
        self.encoder_pos = SoftPositionEmbed(self.last_dim, resolution)

    def forward(self, x):
        x = self.encoder_sa(x)
        x = x.permute(0,2,3,1) # B, W, H, C -> put C in last dimension
        x = self.encoder_pos(x)                 
        # spatial flatten -> flatten the spatial dimensions
        x = torch.flatten(x, 1, 2) # B, W*H, C -> 64 features of size w*H
        return x

class SlotAttentionDecoder(nn.Module):
    def __init__(self, ch_dim, hid_dim, resolution, init_resolution):
        super().__init__()

        self.conv_list = nn.ModuleList(
            [nn.ConvTranspose2d(hid_dim, ch_dim, 4, 2, 1), nn.ELU()]
        )
        for i in range(2):
            self.conv_list.append(nn.ConvTranspose2d(ch_dim, ch_dim, 4, 2, 1))
            self.conv_list.append(nn.ELU())

        self.conv_list.append(nn.ConvTranspose2d(ch_dim, 4, 3, 1, 1))

        self.decoder_pos = SoftPositionEmbed(hid_dim, init_resolution)
        self.resolution = resolution

    def forward(self, x):
        x = self.decoder_pos(x)
        #logger.info(x.shape)
        x = x.permute(0, 3, 1, 2)
        for conv_layer in self.conv_list:
            x = conv_layer(x)
            #logger.info(x.shape)

        x = x[:, :, : self.resolution[0], : self.resolution[1]]
        x = x.permute(0, 2, 3, 1)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, num_slots, slot_dim, init_dim=64, num_iterations=3, resolution=(128, 128), init_decoder_res=(32, 32), softmax=True):
        super().__init__()
    
        self.encoder = Encoder(resolution)
        self.decoder = SlotAttentionDecoder(ch_dim=init_dim, hid_dim=slot_dim, resolution=resolution, init_resolution=init_decoder_res)
        self.num_slots = num_slots

        self.slot_attention = SlotAttention(
                num_slots=num_slots,
                slot_dim=slot_dim,
                dim=init_dim,
                iters=num_iterations,
                eps = 1e-8, 
                hidden_dim=128)
        
        self.mlp = nn.Sequential(
             nn.Linear(init_dim, init_dim), # used to be hid_dim, hid_dim
             nn.ELU(),
             nn.Linear(init_dim, init_dim)
        )
        
        self.spatial_broadcast = init_decoder_res[0]
        self.use_softmax = softmax
        
    def encode(self, x):
        # `x` is an image which has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder(x)  # CNN Backbone.
        #logger.info(f"1: {x.shape}")
        x = F.layer_norm(x, x.shape[1:])

        x = self.mlp(x)

        # Slot Attention module.
        slots, attn = self.slot_attention(x)
        return slots, attn
    
    def decode(self, hat_z):
        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        slots = (
            hat_z.reshape((-1, hat_z.shape[-1]))
            .unsqueeze(1)
            .unsqueeze(2)
            .repeat((1, self.spatial_broadcast, self.spatial_broadcast, 1))
        )

        #logger.info(f"7 (slots broadcasted): {slots.shape}")

        # `out` has shape: [batch_size*num_slots, width, height, num_channels+1].
        out = self.decoder(slots)
        
        #logger.info(f"8 (decoder out): {out.shape}")

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


    def forward(self, x):
        slots, attn = self.encode(x)
        hat_x, figures, masks, reconstructions = self.decode(slots)

        output_dict = {
            "attn": attn,
            "reconstructed_image": hat_x,
            "predicted_latents": slots,
            "reconstructed_figures": figures.permute(1, 0, 4, 2, 3),
            "reconstructed_masks": masks.permute(1, 0, 4, 2, 3),
            "raw_figures": reconstructions.permute(1, 0, 4, 2, 3),
        }
        return output_dict
        


    
