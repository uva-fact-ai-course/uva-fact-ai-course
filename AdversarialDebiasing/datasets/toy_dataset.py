from torch.utils import data
import torch
import numpy as np


class ToyDataset(data.Dataset):
    """
    Toy dataset as described in Zhang et al., Mitigating Unwanted Biases with Adversarial Learning

    Args:
        n_examples (int): Number of examples to generate
    """

    def __init__(self, n_examples):
        self.n_examples = n_examples
        self.r = np.random.choice([0, 1], size=(n_examples, 1))
        self.v = np.random.randn(n_examples, 1) + self.r
        self.u = np.random.randn(n_examples, 1) + self.v
        self.w = np.random.randn(n_examples, 1) + self.v
        self.x = torch.tensor(np.hstack([self.r, self.u])).float()
        self.y = (torch.tensor(self.w) > 0).float() 
        self.z = torch.tensor(self.r).float()

    def __len__(self):
        return self.n_examples

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index]
