import torch
import torch.nn as nn

from contextlib import nullcontext
from typing import Any, Dict

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