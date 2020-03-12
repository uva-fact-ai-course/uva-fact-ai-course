import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


class AdultUCI(Dataset):
    """
    UCI Adult dataset from https://archive.ics.uci.edu/ml/datasets/Adult. Pre-processing of the data is done according to
    Zhang et al., Mitigating Unwanted Biases with Adversarial Learning. The protected variable is sex.

    Args:
        file_paths (list of str): File paths of the training and the test set
    """

    def __init__(self, file_paths):

        self.var_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                          'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                          'native-country', 'income']
        self.real_var_names = ['fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        self.data = []
        self.labels = []
        self.protected = []
        self.encoder = {}

        li = []
        self.lengths = []
        for fp in file_paths:
            with open(fp, 'r') as data:
                li.append(pd.read_csv(data, names=self.var_names))
                self.lengths.append(li[-1].shape[0])

        print(li[0].groupby(['sex', 'income']).income.count())
        print(li[1].groupby(['sex', 'income']).income.count())

        self.data = pd.concat(li, axis=0, ignore_index=True)
        self.data['income'] = self.data['income'].apply(self.clean)

        self.process_data()

    def clean(self, x):
        if isinstance(x, str):
            x = x.replace('.', '')
        return x

    def process_data(self):
        self.data['age'] = pd.cut(self.data['age'],
                                  bins=[0, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 100],
                                  labels=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

        for idx, name in enumerate(self.var_names):
            if name not in self.real_var_names:
                self.encoder[name], one_hot = self.one_hot_encode(self.data[name])
                if name is 'income':
                    self.labels = torch.tensor(self.data[name] == ' >50K').float().unsqueeze(dim=1)
                    continue
                elif name is 'sex':
                    self.protected = torch.tensor(self.data[name] == ' Male').float().unsqueeze(dim=1)
                    continue
                if idx == 0:
                    data_temp = one_hot
                    continue
                else:
                    data_temp = np.append(data_temp, one_hot, axis=1)
            else:
                if idx == 0:
                    data_temp = np.expand_dims(self.data[name].values, axis=1)
                    continue
                elif name is 'fnlwgt':
                    continue
                else:
                    data_temp = np.append(data_temp, np.expand_dims(self.data[name].values, axis=1), axis=1)

        self.data = torch.tensor(data_temp).float()

    def one_hot_encode(self, data):
        encoder = LabelEncoder().fit(data)
        one_hot_idx = encoder.transform(data)
        one_hot = np.eye(len(encoder.classes_))[one_hot_idx]

        return encoder, one_hot

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.protected[idx]
