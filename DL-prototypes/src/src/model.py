"""
Defines two models: PrototypeModel and HierarchyModel.
This is the model that will eventually be trained.
It consist of an autoencoder and a (hierarchical) prototype network
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from src.network.modules import ConvDecoder, ConvEncoder
from src.network.prototype import PrototypeClassifier
from src.network.hierarchyPrototype import HierarchyPrototypeClassifier

class PrototypeModel(nn.Module):
    """
    Defines the entire model -> Encoder, Decoder, and hierarchical prototype network
    TODO: Possibly generalize this where you can give the Encoder and Decoder as arguments somehow?
    """
    def __init__(self, n_prototypes, latent_size, n_classes):
        super().__init__()

        self.encoder = ConvEncoder(latent_size)
        self.decoder = ConvDecoder(latent_size)
        self.prototype = PrototypeClassifier(n_prototypes, latent_size, n_classes)

    def forward(self, x):
        """
        Performs one forward pass of the full model
        Args:
            Input:
                x: batch of data, appropriately sized for the specific encoder and decoder.
            Output:
                encoded : encoded batch of data of size (batch_size, latent_size)
                decoded : decoded batch of data of appropriate input size
                prototype : tuple of (distances, logits). See PrototypeClassifier
        """
        encoded = self.encoder.forward(x)         # f(x)
        decoded = self.decoder.forward(encoded)   # g(f(x))
        prototype = self.prototype.forward(encoded) # h(f(x))

        return encoded, decoded, prototype

class HierarchyModel(nn.Module):
    """
    Defines the entire model -> Encoder, Decoder, and hierarchical prototype network
    TODO: Possibly generalize this where you can give the Encoder and Decoder as arguments somehow?
    """
    def __init__(self, n_prototypes, latent_size, n_classes, n_sub_prototypes):
        super().__init__()

        self.encoder = ConvEncoder(latent_size)
        self.decoder = ConvDecoder(latent_size)
        self.prototype = HierarchyPrototypeClassifier(
            n_prototypes,
            latent_size,
            n_classes,
            n_sub_prototypes)

    def forward(self, x):
        """
        Performs one forward pass of the full model
        Args:
            Input:
                x: batch of data, appropriately sized for the specific encoder and decoder.
            Output:
                encoded : encoded batch of data of size (batch_size, latent_size)
                decoded : decoded batch of data of appropriate input size
                prototype : tuple of (distances, logits). See PrototypeClassifier
        """
        encoded = self.encoder.forward(x)           # f(x)
        decoded = self.decoder.forward(encoded)     # g(f(x))
        prototype = self.prototype.forward(encoded) # h(f(x))

        return encoded, decoded, prototype
