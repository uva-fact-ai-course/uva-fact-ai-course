"""
Reimplementation of prototype network by Li et al.
"""
import torch
import torch.nn as nn
from src.helper import list_of_distances

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PrototypeClassifier(nn.Module):
    """
    Define a nonhierarchical prototype network
    """
    def __init__(self, n_prototypes, latent_size, output_size):
        """
        Initialize prototype module with n_prototypes prototypes in space latent_size.
        Map to output_size classes.
        """
        super().__init__()

        self.latent_size = latent_size
        # initialize n_prototypes prototypes, they are of size latent_size
        self.prototypes = nn.Parameter(
            torch.nn.init.uniform_(torch.zeros(n_prototypes, latent_size)))

        # Initialize Linear Layer
        if n_prototypes == output_size:
            self.linear1 = nn.Linear(n_prototypes, output_size)
            self.linear1.weight.data.copy_(-torch.eye(n_prototypes))

            # Set linear layer parameters to no grad so negative identity remains intact
            for param in self.linear1.parameters():
                param.requires_grad = False

        # If n_sup_prototypes is different from number of classes, make linear1 learnable
        else:
            self.linear1 = nn.Linear(n_prototypes, output_size)


    def forward(self, data_in):
        """
        Perform forward pass for the prototype network.
        Args:
            Input:
                data_in : Latent space encodings of shape (batch_size, latent_size)
                        Latent_size can be any amount of dimensions which multiply to latent_size

            Output:
                x : matrix of distances in latent space of shape (batch_size, n_prototypes)
                out : non-normalized logits for every data point
                      in the batch, shape (batch_size, output_size)
        """
        data_in = data_in.float()
        data_in = data_in.view(len(data_in), self.latent_size)
        distances = list_of_distances(data_in, self.prototypes)
        out = self.linear1(distances)
        # regularization r1: Be close to at least one training example
        # (get min distance to each datapoint=dimension 0)
        r_1 = torch.mean(torch.min(distances, axis=0).values)
        # regularization r2: Be close to at least one prototype
        # (get min distance to each prototype=dimension 1)
        r_2 = torch.mean(torch.min(distances, axis=1).values)
        return r_1, r_2, out

    def get_prototypes(self):
        """
        Get prototypes
        """
        return self.prototypes
