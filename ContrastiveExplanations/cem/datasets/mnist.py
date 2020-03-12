""" This module implements the MNIST class to handle importing
the MNIST dataset.

"""

import torch
import torchvision.datasets as datasets
from torchvision import transforms

from .dataset import Dataset


class MNIST(Dataset):
    def __init__(self, download=False, batch_size=4, shuffle=True, binary=True):
        """ Initialise the MNIST dataset class.

        The MNIST class extends the Dataset class to quickly
        retrieve samples and batches in a way that's the same for
        all datasets.

        download
            download the dataset from the internet
        batch_size
            the batch size of the dataloader
        shuffle
            shuffle the samples in the dataloader
        """

        transform = transforms.Compose([transforms.ToTensor()])

        # Load the test and train raw data, and possibly download it.
        self.train_data = datasets.MNIST(
            root="./cem/datasets/data/mnist",
            train=True,
            download=download,
            transform=transform,
        )
        self.test_data = datasets.MNIST(
            root="./cem/datasets/data/mnist",
            train=False,
            download=download,
            transform=transform,
        )
        # Create a shuffled dataloader for the test and the train set,
        # with a specified batch size.
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=batch_size, shuffle=shuffle
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data, batch_size=batch_size, shuffle=shuffle
        )
        # Create a non-shuffled dataloader with a batch size of 1 for
        # the train and test set.
        self.train_list = torch.utils.data.DataLoader(
            self.train_data, batch_size=1, shuffle=False
        )
        self.test_list = torch.utils.data.DataLoader(
            self.test_data, batch_size=1, shuffle=False
        )
