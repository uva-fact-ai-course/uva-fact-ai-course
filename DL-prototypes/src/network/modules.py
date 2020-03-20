"""
Defines the Encoder and Decoder for the MNIST network specified in Li et al.
"""
import torch
import torch.nn as nn

# Port device to GPU if available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Encoder class for convolutional neural network
class ConvEncoder(nn.Module):
    """
    Encoder network with four convolutional layers.
    Encodes 28*28*1 images to size latent_size
    """
    def __init__(self, latent_size):
        super().__init__()

        # Apparently padding=1 was necessary to get the same dimensions as listed in the paper.
        # There should be a way to do this automatically, like in tensorflow.
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(32, int(latent_size/4), kernel_size=3, stride=2, padding=1),
            nn.Sigmoid())

    def forward(self, data_in):
        """
        Perform forward pass of CNN encoder
        Args:
            Input:
                data_in : MNIST sized image of shape (batch_size, 1, 28, 28)

            Output:
                out : Encoded data of shape  (batch_size * 10 * 2 * 2)
        """
        # Perform full pass through network
        out = self.convnet(data_in)
        return out

class ConvDecoder(nn.Module):
    """
    Decoder with four convolutional layers.
    Decodes latent_size inputs to 28*28*1 images.
    """
    def __init__(self, latent_size):
        super().__init__()

        self.de1 = nn.ConvTranspose2d(int(latent_size/4), 32, kernel_size=3, stride=2, padding=1)
        self.de2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.de3 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.de4 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data_in):
        """
        Perform forward pass of the CNN decoder
        Args:
            Input:
                data_in : Encoded data of shape (batch_size * 10 * 2 * 2)
            Output:
                out : Decoded data of shape (batch_size * 1 * 28 * 28)
        """
        # Output_size is necessary during convolution, to ensure correct dimensionsality.
        # In tensorflow, this can again be done with 'same' padding.
        data_length = len(data_in)
        out = self.de1(data_in, output_size=(data_length, 32, 4, 4))
        out = self.sigmoid(out)
        out = self.de2(out, output_size=(data_length, 32, 7, 7))
        out = self.sigmoid(out)
        out = self.de3(out, output_size=(data_length, 32, 14, 14))
        out = self.sigmoid(out)
        out = self.de4(out, output_size=(data_length, 1, 28, 28))
        out = self.sigmoid(out)
        return out
