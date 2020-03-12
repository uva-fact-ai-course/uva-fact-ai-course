import math
import os
import torch
import gdown
import tarfile
import pickle

from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import MaxAbsScaler

from datasets.adult_dataset import AdultUCI
from datasets.communities_crime_dataset import CommunitiesCrimeDataset
from datasets.image_dataset import UTKFace


def get_utkface_dataloaders(base_path, batch_size):
    """
    Retrieves the train, validation and test dataloaders for the UTKFace dataset.

    Args:
        base_path (str): Home path of the repository
        batch_size (int): Batch size for the dataloaders

    Returns: (train dataloder, validation dataloader, test dataloader)
    """
    data_path = os.path.join(base_path, 'data/UTKFace')

    if not os.path.isdir(data_path):
        # Download dataset if not present
        url = 'https://drive.google.com/uc?id=0BxYys69jI14kYVM3aVhKS1VhRUk'
        out = os.path.join(base_path, 'data/UTKFace.tar.gz')
        gdown.download(url, out, quiet=False, proxy=None)
        with tarfile.open(out, "r:gz") as tar_ref:
            tar_ref.extractall(os.path.join(base_path, 'data'))

    data = UTKFace(data_path, protected_vars=['sex'])

    # Split data into train, validation and test set with 60/20/20 ratio
    train_data, test_data = torch.utils.data.random_split(data, [math.ceil(len(data) * 0.6),
                                                                 len(data) - math.ceil(len(data) * 0.6)])

    test_data, val_data = torch.utils.data.random_split(test_data, [math.ceil(len(test_data) * 0.5),
                                                                    len(test_data) - math.ceil(len(test_data) * 0.5)])

    for no, data in enumerate(['train_data.pkl', 'val_data.pkl', 'test_data.pkl']):
        with open(data, 'wb') as file:
            pickle.dump([train_data.indices, val_data.indices, test_data.indices][no], file)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def load_communities_crime_dataset(base_path, batch_size):
    """
    Retrieves the train and test dataloaders for the Communities and Crime dataset.

    Args:
        base_path (str): Home path of the repository
        batch_size (int): Batch size for the dataloaders

    Returns: (train dataloder, test dataloader)
    """
    cc_dataset = CommunitiesCrimeDataset(os.path.join(base_path,'data/'))

    # Split data into train, validation and test set with 60/20/20 ratio
    end_of_train = int(0.6 * len(cc_dataset))
    end_of_val = end_of_train + int(0.2 * len(cc_dataset))
    train_dataset = Subset(cc_dataset, range(0, end_of_train))
    val_dataset = Subset(cc_dataset, range(end_of_train, end_of_val))
    test_dataset = Subset(cc_dataset, range(end_of_val, len(cc_dataset)))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def load_adult_dataset(base_path, batch_size):
    """
    Retrieves the train and test dataloaders for the Adult dataset.

    Args:
        base_path (str): Home path of the repository
        batch_size (int): Batch size for the dataloaders

    Returns: (train dataloder, test dataloader)
    """
    train_path = os.path.join(base_path, 'data/adult.data')
    test_path = os.path.join(base_path, 'data/adult.test')
    adult_dataset = AdultUCI([train_path, test_path])

    # Split data into train and test set with 60/30 ratio
    end_of_train = int(0.7 * len(adult_dataset))
    min_max_scaler = MaxAbsScaler()
    adult_dataset.data[:end_of_train] = torch.tensor(min_max_scaler.fit_transform(adult_dataset.data[:end_of_train].numpy()))
    adult_dataset.data[end_of_train:] = torch.tensor(min_max_scaler.transform(adult_dataset.data[end_of_train:].numpy()))
    train_dataset = Subset(adult_dataset, range(0, end_of_train))
    test_dataset = Subset(adult_dataset, range(end_of_train, len(adult_dataset)))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    return train_loader, test_loader


def get_dataloaders(batch_size, dataset):
    """
    Gets the relevant dataloaders based on the requested dataset. Validation dataloader is None for the Adult and
    the Communities and Crime datasets.s

    Args:
        batch_size (int): Batch size for the dataloaders
        dataset (str): The name of the dataset - one of {adult, crime, images}

    Returns: (train dataloader, validation dataloader, test dataloader)
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if dataset == 'images':
        train_loader, val_loader, test_loader = get_utkface_dataloaders(base_path, batch_size)
    elif dataset == 'adult':
        train_loader, test_loader = load_adult_dataset(base_path, batch_size)
        val_loader = None
    elif dataset == 'crime':
        train_loader, val_loader, test_loader = load_communities_crime_dataset(base_path, batch_size)
    else:
        raise Exception('Unsupported dataset')

    return train_loader, val_loader, test_loader
