import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from torchvision import transforms


class UTKFace(Dataset):
    """
    UTKFace dataset from https://susanqq.github.io/UTKFace/.
    Pre-processing of the data includes resizing of images, and binning and binarization of age.
    The protected variable is sex.


    Args:
        directory (list of str): File path to the folder containing images
        protected_vars (list): list of strings of protected variable names
    """

    def __init__(self, directory, protected_vars):
        self.samples = []
        self.vars = {'sex': [], 'race': []}
        self.labels = []
        self.protected_var_names = protected_vars

        # We resize the input from 200x200 to 100x100
        self.transform = transforms.Compose([transforms.Resize(100), transforms.ToTensor()])
        skipped = 0

        for idx, imgdir in enumerate(os.listdir(directory)):
            if len(imgdir.split('_')) < 4:
                # Some images are skipped if they meet any of the categorical features
                skipped += 1
                continue
            # Instead of storing the images we store the paths to them and retrieve when calling a batch
            self.samples.append(directory + '/' + imgdir)
            self.labels.append(int(imgdir.split('_')[0]))
            self.vars['sex'].append(int(imgdir.split('_')[1]))
            self.vars['race'].append(int(imgdir.split('_')[2]))
        print(f'Skipped {skipped} images.')


        _, self.labels = self.one_hot_encode(pd.cut(self.labels,
                                                    bins=[0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 120],
                                                    right=True))

        self.labels = torch.tensor(self.labels).float()

        for no, var in enumerate(self.vars):
            if var in self.protected_var_names:
                # 0 is male
                self.protected_vars = torch.tensor([value == 0 for value in self.vars[var]]).float().unsqueeze(dim=1)

    def one_hot_encode(self, data):
        encoder = LabelEncoder().fit(data)
        one_hot_idx = encoder.transform(data)
        one_hot = np.eye(len(encoder.classes_))[one_hot_idx]

        return encoder, one_hot

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with open(self.samples[idx], 'rb') as img:
            image = self.transform(Image.open(img).convert('RGB'))

        return image, self.labels[idx], self.protected_vars[idx]
