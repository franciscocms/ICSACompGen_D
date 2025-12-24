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
from src.models.architectures import SlotMLPAdditive, NoiseEncoder, CausalDecoder, TCDiscriminator, MultivariateCausalFlow, PriorMultivariateCausalFlow

import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(params["running_type"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



"""Slot Attention-based auto-encoder for object discovery."""
class InvSlotAttentionGuide(nn.Module):
    def __init__(self, resolution, num_iterations, latent_dim, nlatents, A, stage):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.resolution = resolution
        self.num_slots = 0
        self.num_iterations = num_iterations
        self.stage = stage
        assert self.stage in ["train", "eval"], "stage must be either 'train' or 'eval'"
        self.L = nlatents
        self.A = A
        self.ldim = latent_dim
        self.slot_dim = self.L*self.ldim

        self.model = SlotMLPAdditive(
            in_channels = 3,
            n_slots = 2,
            n_slot_latents = 16,
        ).to(device)


        self.noise_nets = nn.ModuleDict()
        for var_name in params["latent_vars"]:
            self.noise_nets[var_name] = NoiseEncoder(nin = self.slot_dim,
                                            nh = self.ldim,
                                            nout = self.ldim
                                            )
        self.causal_decoder = CausalDecoder(nin = self.slot_dim,
                                        nh = self.slot_dim,
                                        nout = self.slot_dim
                                        )
        
        self.causal_flow = MultivariateCausalFlow(self.L, self.ldim, self.A)
        self.prior = PriorMultivariateCausalFlow(self.L, self.ldim, self.A)
        if params["TC_minimization"]: self.tc_discriminator = TCDiscriminator(self.L*self.ldim, self.L*self.ldim*4)
        self.epoch = 0
        
    
    def forward(self, img, labels):

        B, _, _, _ = img.shape
        n_s = params['num_slots'] 

        assert labels.shape == (B, n_s, self.L), f"{labels.shape}"

        output_dict = self.model(img)
        img_hat = output_dict["reconstructed_image"]
        recon_figures = output_dict["reconstructed_figures"]
        slots = output_dict["predicted_latents"]

        slots = slots.detach()

        z_noise, mu_noise, logvar_noise = [], [], []           
        for var in params["latent_vars"]:
            mu, logvars, e = self.noise_nets[var].encode(slots)
            z_noise.append(e)
            mu_noise.append(mu)
            logvar_noise.append(logvars) 
        z_noise = torch.stack(z_noise).permute(1, 2, 0, 3)            # [B, N, L, d] => 'L' refers to the number of categorical latents
        mu_noise = torch.stack(mu_noise).permute(1, 2, 0, 3)          # [B, N, L, d]
        logvar_noise = torch.stack(logvar_noise).permute(1, 2, 0, 3)  # [B, N, L, d]
        assert z_noise.size(2) == self.L
        
        u_m = torch.ones(B, n_s, self.L, self.ldim).to(device)
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

        slot_hat = self.causal_decoder(torch.flatten(z_scm, -2))
        
        if params["beta"] > 0:
            cp, self.prior_L_slopes, prior_ss, prior_ts = self.prior(u_m, z_scm) # [B, N, L, d]
            self.prior_ss, self.prior_ts = prior_ss, prior_ts     
                            
        if params["running_type"] == "eval": return z_noise.reshape(B, n_s, self.L, self.ldim)
        return slots, img_hat, z_noise, mu_noise, logvar_noise, z_scm, slot_hat, cp
        





        # self.img = img
        # x = self.encoder_cnn(self.img)
        # x = nn.LayerNorm(x.shape[1:]).to(device)(x)
        # self.features_to_slots = self.mlp(x)
        # slots, self.slot_pos, attn = self.slot_attention(self.features_to_slots, num_slots=n_s) # slots in [B, n_s, d]
        # self.attn = attn
                                
        # z_noise, mu_noise, logvar_noise, rec_noise = [], [], [], []           
        # for var in params["latent_vars"]:
        #     mu, logvars, e = self.noise_nets[var].encode(slots)
        #     z_noise.append(e)
        #     mu_noise.append(mu)
        #     logvar_noise.append(logvars) 
        # z_noise = torch.stack(z_noise).permute(1, 2, 0, 3)            # [B, N, L, d] => 'L' refers to the number of categorical latents
        # mu_noise = torch.stack(mu_noise).permute(1, 2, 0, 3)          # [B, N, L, d]
        # logvar_noise = torch.stack(logvar_noise).permute(1, 2, 0, 3)  # [B, N, L, d]
        # assert z_noise.size(2) == self.L
        
        # u_m = torch.ones(B, n_s, self.L, self.ldim).to(device)
        # for i, var in enumerate(params["latent_vars"]):
        #     u_m[:, :, i] = ((labels[:, :, i] - params["factor_ranges"][var][0]) / (params["factor_ranges"][var][1] - params["factor_ranges"][var][0])).unsqueeze(-1).expand(-1, -1, self.ldim)
        
        # if params["causal_dropout_prob"] <= 1.:
        #     z_scm, self.L_slopes, ss, ts = self.causal_flow(z_noise)
        #     self.ss, self.ts = ss, ts
        # else:
        #     z_scm = z_noise 

        # z_noise = z_noise.reshape(B*n_s, self.L, self.ldim)
        # mu_noise = mu_noise.reshape(B*n_s, self.L, self.ldim)
        # logvar_noise = logvar_noise.reshape(B*n_s, self.L, self.ldim)
        # self.logvar_noise = logvar_noise

        
        # slot_hat = []
        # for i in range(n_s):
        #     slot_hat_i = self.causal_decs[str(i)](z_scm[:, i, :, :])
        #     #logger.info(slot_hat_i.shape)
        #     slot_hat.append(slot_hat_i) # N, B, L, d
        # slot_hat = torch.sum(torch.stack(slot_hat, dim=1), dim=-2) # [B, N, L, d] -> [B, N, d]
        
        # img_hat = self.decoder(slots)

        # """
        # - prior won't be aligned with labels...
        # """

        # if params["beta"] > 0:
        #     cp, self.prior_L_slopes, prior_ss, prior_ts = self.prior(u_m, z_scm) # [B, N, L, d]
        #     self.prior_ss, self.prior_ts = prior_ss, prior_ts     
                            
        # if self.stage == "eval": return z_noise.reshape(B, n_s, self.L, self.ldim)
        # return slots, img_hat, z_noise, mu_noise, logvar_noise, z_scm, slot_hat, cp
        
        
        
        
        
        

