import torch
import torch.nn as nn


class Predictor(nn.Module):
    """
    Basic predictor model for binary output.

    Args:
        input_dim (int): Number of input dimensions
    """
    def __init__(self, input_dim):
        super(Predictor, self).__init__()
        self.linear_layer = nn.Linear(input_dim, 1)
        self.output_layer = nn.Sigmoid()

    def forward(self, x):
        logits = self.linear_layer(x)
        predictions = self.output_layer(logits)
        return logits, predictions


class ImagePredictor(nn.Module):
    """
    Convolutional predictor model for images.

    Args:
        input_dim (int): Number of input dimensions
        output_dim (int): Number of output dimensions
        drop_prob (float, default 0.3): Probability for dropout
    """
    def __init__(self, input_dim, output_dim, drop_prob=0.3):
        super(ImagePredictor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_probability = drop_prob
        self.convolutions = nn.Sequential(
            nn.Conv2d(self.input_dim, 6, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0),
            nn.Conv2d(6, 12, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0),
            nn.Dropout(drop_prob),
            nn.Conv2d(12, 24, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0),
            nn.Dropout(drop_prob))
        self.linears = nn.Sequential(
            nn.Linear(24 * 9 * 9, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        convoluted = self.convolutions(x)
        logits = self.linears(convoluted.view(convoluted.size(0), -1))
        preds = self.sigmoid(logits)
        return logits, preds


class Adversary(nn.Module):
    """
    Adversary model as described in Zhang et al., Mitigating Unwanted Biases with Adversarial Learning.

    Args:
        input_dim (int): Number of input dimensions
        protected_dim (int): Number of dimensions for the protected variable
    """
    def __init__(self, input_dim, protected_dim, equality_of_odds=True):
        super(Adversary, self).__init__()
        self.c = nn.Parameter(torch.ones(1 * protected_dim))
        if equality_of_odds:
            self.no_targets = 3
        else:
            self.no_targets = 1
        self.w2 = nn.init.xavier_uniform_(nn.Parameter(torch.empty(self.no_targets * input_dim, 1 * protected_dim)))
        self.b = nn.Parameter(torch.zeros(1 * protected_dim))

    def forward(self, logits, targets):
        s = torch.sigmoid((1 + torch.abs(self.c)) * logits)
        if self.no_targets == 3:
            z_hat = torch.cat((s, s * targets, s * (1 - targets)), dim=1) @ self.w2 + self.b
        else:
            z_hat = s @ self.w2 + self.b
        return z_hat, torch.sigmoid(z_hat)
