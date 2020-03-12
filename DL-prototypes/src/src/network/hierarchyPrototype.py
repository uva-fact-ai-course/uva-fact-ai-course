"""
Defines the hierarchy prototype network.
To be attached to the autoencoder.
"""
import torch
import torch.nn as nn
from src.helper import list_of_distances

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HierarchyPrototypeClassifier(nn.Module):
    """
    Hierarchical prototype network
    """

    def __init__(self, n_sup_prototypes, latent_size, output_size, n_sub_prototypes):
        """
        Initialize prototype module with n_sup_prototypes super prototypes in space latent_size.
        Map to output_size classes.
        """
        super().__init__()
        self.n_sup_prototypes = n_sup_prototypes
        self.n_sub_prototpes = n_sub_prototypes
        self.latent_size = latent_size
        self.output_size = output_size
        # initialize n_sup_prototypes super prototypes, they are of size latent_size
        self.sup_prototypes = nn.Parameter(
            torch.nn.init.uniform_(torch.zeros(n_sup_prototypes, latent_size)))
        self.sub_prototypes = nn.Parameter(
            torch.nn.init.uniform_(torch.zeros(n_sub_prototypes, latent_size)))

        # Linear layers for super prototypes and sub prototypes
        if n_sup_prototypes == output_size:
            self.linear1 = nn.Linear(n_sup_prototypes, output_size)
            self.linear1.weight.data.copy_(-torch.eye(n_sup_prototypes))

            # Set linear layer parameters to no grad so negative identity remains intact
            for param in self.linear1.parameters():
                param.requires_grad = False

        # If n_sup_prototypes is different from number of classes, make linear1 learnable
        else:
            self.linear1 = nn.Linear(n_sup_prototypes, output_size)

        self.linear2 = nn.Linear(n_sub_prototypes, output_size)


    def forward(self, data_in):
        """
        Perform forward pass for the prototype network.
        Args:
            Input:
                data_in : Latent space encodings of shape (batch_size, latent_size)
                        latent_size can be any amount of dimensions which multiply to latent_size

            Output:
                x : matrix of distances in latent space of shape (batch_size, n_sup_prototypes)
                out : non-normalized logits for every data point
                      in the batch, shape (batch_size, output_size)
        """
        # Port input to float tensor
        data_in = data_in.float()
        # Latent space is 10x2x2 = 40
        data_in = data_in.view(len(data_in), self.latent_size)

        # Distances between input and prototypes
        sup_input_dist = list_of_distances(data_in, self.sup_prototypes)
        sub_input_dist = list_of_distances(data_in, self.sub_prototypes)

        #Compute unnormalized classification probabilities
        sup_out = self.linear1(sup_input_dist)
        sub_out = self.linear2(sub_input_dist)

        # Clone and detach so r_3 and r_4 do not affect sub_prototype parameters
        sub_clones = self.sub_prototypes.clone()
        sub_clones = sub_clones.detach()
        sub_clones.requires_grad = False
        # Calculate distances for sub- and super prototypes
        super_sub_dist = list_of_distances(self.sup_prototypes, sub_clones)

        # r_1 forces sub proto to be close to one training example
        # r_2 forces one training example to be close to sub proto
        r_1 = torch.mean(torch.min(sub_input_dist, axis=0).values)
        r_2 = torch.mean(torch.min(sub_input_dist, axis=1).values)

        # Forcing sub prototype to look like super prototype and vice versa
        # r_3 forces super prototype to be close to sub prototype
        # r_4 forces sub prototype to be close to super prototype
        r_3 = torch.mean(torch.min(super_sub_dist, axis=1).values)
        r_4 = torch.mean(torch.min(super_sub_dist, axis=0).values)

        return sub_out, sup_out, r_1, r_2, r_3, r_4

    def get_prototypes(self):
        """
        Return superprototypes
        """
        return self.sup_prototypes

    def get_sub_prototypes(self):
        """
        Return subprototypes
        """
        return self.sub_prototypes
