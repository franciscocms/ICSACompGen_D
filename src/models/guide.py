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
from src.models.architectures import SlotMLPAdditive, SlotEncoder, SlotMLPAdditiveDecoder, NoiseEncoder, CausalDecoder, TCDiscriminator, MultivariateCausalFlow, ConditionalGaussianPrior, Predictor
from src.utils.loss import compute_matching_cost

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
        self.slot_dim = self.L*self.ldim*4
        #self.slot_dim = self.L*self.ldim

        self.model = SlotMLPAdditive(
            in_channels = 3,
            n_slots = 2,
            n_slot_latents = self.slot_dim,
        ).to(device)

        # self.slot_encoder = SlotEncoder(in_channels=3, n_slots=2, n_slot_latents=self.slot_dim)
        # self.slot_decoder = SlotMLPAdditiveDecoder(in_channels=3, n_slots=2, n_slot_latents=self.slot_dim)


        self.noise_nets = nn.ModuleDict()
        for var_name in params["latent_vars"]:
            self.noise_nets[var_name] = NoiseEncoder(nin = self.slot_dim,
                                            nh = self.ldim,
                                            nout = self.ldim,
                                            slot_dim = self.slot_dim
                                            )
        self.causal_decoder = CausalDecoder(nin = self.L*self.ldim,
                                        nout = self.slot_dim
                                        )
        
        self.causal_flow = MultivariateCausalFlow(self.L, self.ldim, self.A)
        self.prior = ConditionalGaussianPrior(self.L, self.ldim)
        self.predictor = Predictor(self.slot_dim, self.slot_dim, self.L)
        self.noise_predictor = Predictor(self.L*self.ldim, self.L*self.ldim, self.L)
        
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

        #slots = self.slot_encoder(img)

        slots = slots.detach()

        z_noise, mu_noise, logvar_noise = [], [], []           
        for var in params["latent_vars"]:
            #logvars = torch.zeros(B, n_s, self.ldim).to(device)
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
        
        logger.info(labels[:5])
        logger.info(u_m[:5])
        
        #slots_d = slots.detach()
        preds = self.predictor(slots)
        self.pred_loss, aligned_u_m = compute_matching_cost(preds, labels, u_m)

        # compute the conditionally factorial prior for noise representations
        prior_means = self.prior(aligned_u_m)
        prior_logvars = torch.zeros_like(prior_means).to(device)
        
        # causal flow forward
        if params["causal_dropout_prob"] < 1.:
            assert not torch.isnan(z_noise).any(), f"{z_noise}"
            z_scm, self.L_slopes, ss, ts = self.causal_flow(z_noise)
            self.ss, self.ts = ss, ts
        else:
            z_scm = z_noise 

        #slot_hat = self.causal_decoder(torch.flatten(z_scm, -2))
        slot_hat = self.causal_decoder(torch.flatten(z_noise, -2))
        
        z_noise_d = z_noise.detach()
        noise_preds = self.noise_predictor(z_noise_d.reshape(B, n_s, self.L*self.ldim))
        self.noise_pred_loss, _ = compute_matching_cost(noise_preds, labels, u_m) 
                            
        if params["running_type"] == "eval": return z_noise.reshape(B, n_s, self.L, self.ldim)
        return slots, img_hat, z_noise, mu_noise, logvar_noise, prior_means, prior_logvars, z_scm, slot_hat
        