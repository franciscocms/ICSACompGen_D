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
from src.models.architectures import SlotMLPAdditive, NoiseEncoder, CausalDecoder, TCDiscriminator, MultivariateCausalFlow, ConditionalGaussianPrior, Predictor
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

        if model_name == "slot_attention":
            slot_encoder = SlotAttentionEncoder(resolution=params["resolution"],
                                                hid_dim=self.slot_dim,
                                                ch_dim=32)
            slot_decoder = SlotAttentionDecoder(ch_dim=32,
                                                hid_dim=self.slot_dim,
                                                resolution=params["resolution"],
                                                init_resolution=params["resolution"])
            self.model = SlotAttentionAutoEncoder(encoder=slot_encoder,
                                                    decoder=slot_decoder,
                                                    num_slots=params["num_slots"],
                                                    num_iterations=3,
                                                    hid_dim=self.slot_dim,
                                                    init_resolution=params["resolution"],
                                                    softmax=True,
                                                    sampling=True)
        elif model_name == "additiveAE":
            self.model = SlotMLPAdditive(in_channels=3,
                                         n_slots=params["num_slots"],
                                         n_slot_latents=self.slot_dim,
                                         )
        else:
            raise ValueError(f"{model_name} is not a valid model name!")


        self.noise_nets = nn.ModuleDict()
        self.noise_predictors = nn.ModuleDict()
        for var_name in params["latent_vars"]:
            self.noise_nets[var_name] = NoiseEncoder(nin = self.slot_dim,
                                            nh = self.ldim,
                                            nout = self.ldim,
                                            slot_dim = self.slot_dim
                                            )
            self.noise_predictors[var_name] = Predictor(self.ldim,
                                                        self.ldim,
                                                        1)
            
        
        
        self.causal_decoder = CausalDecoder(nin = self.L*self.ldim,
                                            nout = self.slot_dim
                                            )
        
        self.causal_flow = MultivariateCausalFlow(self.L,
                                                  self.ldim,
                                                  self.A)
        self.prior = ConditionalGaussianPrior(self.L,
                                              self.ldim,
                                              learn_var=True)
        self.predictor = Predictor(self.slot_dim,
                                   self.slot_dim,
                                   self.L)
        
        
        if params["TC_minimization"]: self.tc_discriminator = TCDiscriminator(self.L*self.ldim, self.L*self.ldim*4)
        self.epoch = 0
        self.iter = 0
        self.learn_transform = 0.
        
    
    def forward(self, img, labels):

        B, _, _, _ = img.shape
        n_s = params['num_slots'] 

        assert labels.shape == (B, n_s, self.L), f"{labels.shape}"

        logger.info(f"{img.shape} - {labels.shape}")

        output_dict = self.model(img)
        img_hat = output_dict["reconstructed_image"]
        recon_figures = output_dict["reconstructed_figures"]
        slots = output_dict["predicted_latents"]
        if self.model_name == "slot_attention":
            decoder_masks = output_dict["reconstructed_masks"]
            self.attn = output_dict["attn"]
        else:
            decoder_masks = None

        slots = slots.detach()
        slots = (slots - slots.mean(dim=(0, 1), keepdim=True)) / slots.std(dim=(0, 1), keepdim=True)

        # logger.info(f"mean: {slots.mean()}")
        # logger.info(f"std: {slots.std()}")
        # logger.info(f"min: {slots.min()}")
        # logger.info(f"max: {slots.max()}\n")

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

        if params["running_type"] == "eval": return z_noise.reshape(B, n_s, self.L, self.ldim)

        #slots_d = slots.detach()
        
        logger.info(slots.shape)

        preds = self.predictor(slots)
        self.pred_loss, aligned_u_m = compute_matching_cost(preds, labels)

        # compute the conditionally factorial prior for noise representations
        if self.prior.learn_var:
            prior_means, prior_logvars = self.prior(aligned_u_m)
        else:
            prior_means = self.prior(aligned_u_m)
            prior_logvars = -3. * torch.ones_like(prior_means).to(device)
        
        # causal flow forward
        if self.learn_transform == 1:
            z_scm, self.L_slopes, ss, ts = self.causal_flow(mu_noise)
            self.ss, self.ts = ss, ts
        else:
            z_scm = mu_noise 
            self.L_slopes, self.ss, self.ts = torch.zeros(B, n_s, self.L, self.ldim), torch.zeros(B, n_s, self.L, self.ldim), torch.zeros(B, n_s, self.L, self.ldim)

        # if self.epoch % 10 == 0 and self.iter == 0:
        #     for i in range(self.L):
        #         logger.info(f"\nvar {i}")
        #         logger.info(f"s mean: {torch.mean(torch.mean(self.ss[:, :, i, :], dim=-1))}")
        #         logger.info(f"s std: {torch.mean(torch.std(self.ss[:, :, i, :], dim=-1))}")
        #         logger.info(f"s min: {torch.mean(torch.amin(self.ss[:, :, i, :], dim=-1))}")
        #         logger.info(f"s max: {torch.mean(torch.amax(self.ss[:, :, i, :], dim=-1))}")
        
        
        slot_hat = self.causal_decoder(torch.flatten(z_scm, -2))
        #slot_hat = self.causal_decoder(torch.flatten(mu_noise, -2))

        if self.epoch % 10 == 0 and self.iter == 0:
            logger.info(f"\nslots mean: {torch.mean(torch.mean(slots, dim=-1))}")
            logger.info(f"slot hat mean: {torch.mean(torch.mean(slot_hat, dim=-1))}")

            logger.info(f"\nslots std: {torch.mean(torch.std(slots, dim=-1))}")
            logger.info(f"slot hat std: {torch.mean(torch.std(slot_hat, dim=-1))}")
            
            logger.info(f"\nslots min: {torch.mean(torch.amin(slots, dim=-1))}")
            logger.info(f"slot hat min: {torch.mean(torch.amin(slot_hat, dim=-1))}")

            logger.info(f"\nslots max: {torch.mean(torch.amax(slots, dim=-1))}")
            logger.info(f"slot hat max: {torch.mean(torch.amax(slot_hat, dim=-1))}")
        
        z_noise_d = mu_noise.detach()
        self.noise_pred_loss = []
        for i, var in enumerate(params["latent_vars"]):
            noise_preds = self.noise_predictors[var](z_noise_d[:, :, i, :])
            if noise_preds.size(-1) == 1: noise_preds = noise_preds.squeeze(-1)
            noise_pred_loss, _ = compute_matching_cost(noise_preds, labels[:, :, i]) 
            self.noise_pred_loss.append(noise_pred_loss)
        self.noise_pred_loss = torch.stack(self.noise_pred_loss)
        
        return slots, img_hat, z_noise, mu_noise, logvar_noise, prior_means, prior_logvars, z_scm, slot_hat, recon_figures, decoder_masks
        #return slots, img_hat, recon_figures, decoder_masks
        