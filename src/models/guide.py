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
from src.models.architectures import SlotMLPAdditive, SlotMLPAdditiveDecoder, FullNoiseEncoder, CausalDecoder, TCDiscriminator, MultivariateCausalFlow, ConditionalGaussianPrior, Predictor, PriorMultivariateCausalFlow
from src.models.slot_attention import SlotAttentionAutoEncoder, SlotAttentionEncoder, SlotAttentionDecoder
from src.utils.loss import compute_matching_cost

import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(params["running_type"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



"""Slot Attention-based auto-encoder for object discovery."""
class InvSlotAttentionGuide(nn.Module):
    def __init__(self, model_name, resolution, num_iterations, latent_dim, nlatents, A):
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
        self.L = nlatents
        self.A = A
        self.ldim = latent_dim
        #self.slot_dim = self.L*self.ldim*4
        self.slot_dim = self.L*self.ldim
        self.model_name = model_name

        # if model_name == "slot_attention":
        #     slot_encoder = SlotAttentionEncoder(resolution=params["resolution"],
        #                                         hid_dim=self.slot_dim,
        #                                         ch_dim=32)
        #     slot_decoder = SlotAttentionDecoder(ch_dim=32,
        #                                         hid_dim=self.slot_dim,
        #                                         resolution=params["resolution"],
        #                                         init_resolution=params["resolution"])
        #     self.model = SlotAttentionAutoEncoder(encoder=slot_encoder,
        #                                             decoder=slot_decoder,
        #                                             num_slots=params["num_slots"],
        #                                             num_iterations=3,
        #                                             hid_dim=self.slot_dim,
        #                                             init_resolution=params["resolution"],
        #                                             softmax=True,
        #                                             sampling=True)
        # elif model_name == "additiveAE":
        #     self.model = SlotMLPAdditive(in_channels=3,
        #                                  n_slots=params["num_slots"],
        #                                  n_slot_latents=self.slot_dim,
        #                                  )
        # else:
        #     raise ValueError(f"{model_name} is not a valid model name!")

        


        
        self.noise_predictors = nn.ModuleDict()
        self.flatt_noise_predictor = nn.ModuleDict()
        for var_name in params["latent_vars"]:
            self.noise_predictors[var_name] = Predictor(self.ldim,
                                                        self.ldim,
                                                        1)
        
            self.flatt_noise_predictor[var_name] = Predictor(self.L*self.ldim, self.L*self.ldim, 1)
            
        self.noise_encoder = FullNoiseEncoder(in_channels=3,
                                              n_slots=params["num_slots"],
                                              n_latents=self.L,
                                              latent_dim=self.ldim,
                                              stage=params["running_type"])
        
        self.scene_decoder = SlotMLPAdditiveDecoder(in_channels=3,
                                                    n_slots=params["num_slots"],
                                                    n_slot_latents=self.slot_dim
                                                    )
        
        self.causal_decoder = CausalDecoder(nin = self.ldim,
                                            nout = self.slot_dim
                                            )
        self.causal_flow = MultivariateCausalFlow(self.L,
                                                  self.ldim,
                                                  self.A,
                                                  stage=params["running_type"])
        self.noise_prior = ConditionalGaussianPrior(self.L,
                                                    self.ldim,
                                                    learn_var=True,
                                                    clamp_var=True)
        
        self.prior = PriorMultivariateCausalFlow(self.ldim)

        self.predictor = Predictor(self.slot_dim,
                                   self.slot_dim,
                                   self.L)
        
        
        if params["TC_minimization"]: self.tc_discriminator = TCDiscriminator(self.L*self.ldim, self.L*self.ldim*4)
        self.epoch = 0
        self.iter = 0
        
    
    def forward(self, img, labels, step=0):

        B, _, _, _ = img.shape
        n_s = params['num_slots'] 

        assert labels.shape == (B, n_s, self.L), f"{labels.shape}"

        # output_dict = self.model(img)
        # img_hat = output_dict["reconstructed_image"]
        # recon_figures = output_dict["reconstructed_figures"]
        # slots = output_dict["predicted_latents"]
        # if self.model_name == "slot_attention":
        #     decoder_masks = output_dict["reconstructed_masks"]
        #     self.attn = output_dict["attn"]
        # else:
        #     decoder_masks = None


        # regress attributes from slots to get an aligned version of labels
        # preds = self.predictor(slots_d)
        # self.pred_loss, aligned_u_m = compute_matching_cost(preds, labels)

        mu_noise, logvar_noise, z_noise = self.noise_encoder(img) # [B, N, L*ldim]
        
        # get initial predictions and aligned labels
        mu_noise_d = mu_noise.detach()
        preds = self.predictor(mu_noise_d)
        self.pred_loss, aligned_u_m = compute_matching_cost(preds, labels)
        
        # get latent-wise noise distributions
        mu_noise = mu_noise.reshape(mu_noise.size(0), mu_noise.size(1), self.L, self.ldim)
        logvar_noise = logvar_noise.reshape(logvar_noise.size(0), logvar_noise.size(1), self.L, self.ldim)
        z_noise = z_noise.reshape(z_noise.size(0), z_noise.size(1), self.L, self.ldim)       
        
        # causal flow forward
        z_scm, ss, ts, log_det, int_flag = self.causal_flow(z_noise, step)
        self.ss, self.ts, self.log_det, self.int_flag = ss, ts, log_det, int_flag       
        
        # get noise prior conditioned on u
        self.mu_noise_prior, self.logvar_noise_prior = self.noise_prior(aligned_u_m)

        # get the causally disentangled prior p(z|u) to push z representations
        self.prior_z = self.prior(aligned_u_m, z_scm)        
        
        slots = self.causal_decoder(z_scm)
        img_hat, recon_figures = self.scene_decoder(slots)

        if params["running_type"] == "eval": 
            return z_noise, img_hat, recon_figures

        z_noise_d = z_noise.detach()
        self.noise_pred_loss = []
        all_noise_preds, all_flat_noise_preds = [], []
        self.flatt_noise_pred_loss = []
        for i, var in enumerate(params["latent_vars"]):
            noise_preds = self.noise_predictors[var](z_noise_d[:, :, i, :])
            if noise_preds.size(-1) == 1: noise_preds = noise_preds.squeeze(-1)
            noise_pred_loss, _ = compute_matching_cost(noise_preds, labels[:, :, i]) 
            self.noise_pred_loss.append(noise_pred_loss)
            all_noise_preds.append(noise_preds)

            flat_noise_preds = self.flatt_noise_predictor[var](torch.flatten(z_noise_d, -2)).squeeze(-1)
            flatt_noise_pred_loss, _ = compute_matching_cost(flat_noise_preds, labels[:, :, i])
            all_flat_noise_preds.append(flat_noise_preds)
            self.flatt_noise_pred_loss.append(flatt_noise_pred_loss)

        all_noise_preds = torch.stack(all_noise_preds, dim=-1)
        all_flat_noise_preds = torch.stack(all_flat_noise_preds, dim=-1)
        self.noise_pred_loss = torch.stack(self.noise_pred_loss)
        self.flatt_noise_pred_loss = torch.stack(self.flatt_noise_pred_loss)

        return slots, img_hat, z_noise, mu_noise, logvar_noise, z_scm, recon_figures
        