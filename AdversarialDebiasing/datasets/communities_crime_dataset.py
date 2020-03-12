import os

import torch
from torch.utils.data import Dataset
import pandas as pd


class CommunitiesCrimeDataset(Dataset):
    """
    Communities and Crime dataset from http://archive.ics.uci.edu/ml/datasets/communities+and+crime. The task is to
    predict the crime rate which is continuous. The protected variable is the percentage of whites in the population which
    is also continuous.

    Args:
        data_dir (str): Directory containing the data
    """
    def __init__(self, data_dir):
        self.attributes = pd.read_csv(os.path.join(data_dir, 'communities_attributes.csv'),
                                      skipinitialspace=True).columns
        csv_data = pd.read_csv(os.path.join(data_dir, 'communities.data'), sep=',', header=None,
                               names=self.attributes, skipinitialspace=True, na_values=["?"])
        csv_data = csv_data.drop(columns=['communityname', 'state', 'county', 'community', 'fold'])
        csv_data = csv_data.fillna(csv_data.mean())
        self.ground_truth = torch.tensor(csv_data['ViolentCrimesPerPop'].values).float().unsqueeze(dim=1)
        self.protected = torch.tensor(csv_data['racePctWhite']).float().unsqueeze(dim=1)
        self.data = torch.tensor(csv_data.drop(columns=['ViolentCrimesPerPop']).values).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.ground_truth[index], self.protected[index]
