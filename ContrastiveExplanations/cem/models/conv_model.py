""" This module implements a convolutional neural classifier.

"""

import torch.nn as nn


class CNN(nn.Module):

    def __init__(
            self, n_channels=1, n_classes=10,
            conv_kernel=(3, 3), pool_kernel=(2, 2), device='cpu'):
        """
        Initialise the CNN.

        n_channels
            number of input channels
        n_classes
            number of output nodes / classes.
        conv_kernel
            kernel size used in the convolutional layers.
        pool_kernel
            kernel size used in the max pooling layers
        device
            CPU or cuda, device to initialise the cnn on.
        """
        super(CNN, self).__init__()

        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels, out_channels=32,
                kernel_size=conv_kernel, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32,
                kernel_size=conv_kernel, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel)
        ).to(device)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=conv_kernel, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64,
                kernel_size=conv_kernel, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel)
        ).to(device)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=200, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=200, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=n_classes, bias=True)
        ).to(device)

    def forward(self, x):
        """
        Forward pass of the neural network with softmax applied at the
        output layer.

        x
            input sample

        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fc(out)
        out = nn.functional.softmax(out, dim=1)
        return out

    def forward_no_sm(self, x):
        """
        Forward pass of the neural network without softmax applied to the
        output layer. This is used in the optimisation objective of the CEM.

        x
            input sample
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fc(out)

        return out
