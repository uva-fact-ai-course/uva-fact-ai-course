""" This module implements the convolutuional autoencoder used as
reconstruction regulariser for the CEM.

"""
import torch.nn as nn


class CAE(nn.Module):

    def __init__(self, device='cpu'):
        """
        Initialise the convolutional autoencoder.

        device
            cuda or cpu, device to initialise the model on.

        """
        super(CAE, self).__init__()

        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        ).to(device)

        self.decoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        ).to(device)

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        x
            input sample.
        """
        out = self.encoder(x)
        out = self.decoder(out)
        return out
