"""
    This file contains all utility functions for reading and saving Cifar100 datasets.
"""
import os
import torch

from torchvision import datasets, transforms, utils

from .transforms import CIFAR_100_TRANSFORM_TEST, CIFAR_100_TRANSFORM_TRAIN, CIFAR_10_TRANSFORM


def load_imageFolder_data(batch_size, transform, shuffle, num_workers, data_dir):
    """
        Load image with classes in a directory
        data_dir : $projectroot/$data_dir/$dataset_name/$dataset
        example  : ~/fact-full-grad-uva/dataset/cifar-100-imageFolder/train/
                : ~/fact-full-grad-uva/dataset/imagenet/

        This function could be moved as it is more general than just cifar100
    """
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=shuffle, num_workers=num_workers)
    return dataloader

def load_data(batch_size, transform, shuffle, num_workers, data_dir, dataset_name, train=True, name='cifar100'):
    '''
        This function gets the original cifar dataset and processes it to be readable by the 
        imageFolder dataloader.
    '''
    if not os.path.exists(data_dir + '/' + dataset_name):
        print("Downloading dataset and extracting in image folder, please wait")
        prepare_data(data_dir, data_dir + '/' + dataset_name, name=name)
        print('Dataset is prepared')
    
    data_dir = '%s/%s/%s' % (data_dir, dataset_name, 'train' if train else 'test')
    return load_imageFolder_data(batch_size, transform, shuffle, num_workers, data_dir)
    
def save_imagefolder_image(data_dir, label, image, name, dataset):
    """
        This function saves a single cifar image in the correct format.
        data_dir: the root dataset directory, 
        name: the name of the image
        dataset: either train or test
        label: the class of the image
        image: the image to be saved
    """
    utils.save_image(image, '%s/%s/%s/%s.jpg' % (data_dir, dataset, str(label.item()), name))

def save_data_loader_data(data_dir, dataloader, batch_size, dataset):
    """
        this function gets al images from a dataloader and saves them in a cifar100 
        imageFolder structure.
    """
    for idx, (data, label) in enumerate(dataloader):
        save_imagefolder_image(data_dir, label, data, idx, dataset)

def create_imagefolder_dir(dir, num_classes):
    """
        This function creates the outline for a cifar dataset structure.
        dir is the path until the dataset_name: ~/fact-full-grad-uva/dataset/cifar-100-imageFolder/
        This function creates ./train/0-99/ and ./test/0.99/
        where train and test are two datasets, and 0-99 are the classes of the images.
    """
    for dataset in ['train', "test"]:
        for class_idx in range(0, num_classes):
            os.makedirs('%s/%s/%s' % (dir, dataset, str(class_idx)))

def prepare_data(load_dir, save_dir, name='cifar100'):
    """
        This function processes the cifar100 data as it is downloaded and prepares it so
        it can be read by a imageFolder dataloader. 
    """
    batch_size = 1 # more samples is not supported
    if not os.path.exists(save_dir):
        if name == 'cifar10':
            create_imagefolder_dir(save_dir, 10)
        if name == 'cifar100':
            create_imagefolder_dir(save_dir, 100)
        for train in [True, False]:
            
            if name == 'cifar10':
                dataset = datasets.CIFAR10(root=load_dir, train=train, transform=CIFAR_10_TRANSFORM, target_transform=None, download=True)

            elif name == 'cifar100':
                transform = CIFAR_100_TRANSFORM_TRAIN if train else CIFAR_100_TRANSFORM_TEST
                dataset = datasets.CIFAR100(root=load_dir, train=train, transform=transform, download=True)

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

            save_data_loader_data(save_dir, dataloader, batch_size, '/train' if train else '/test')
    else:
        print("Directory already exists, images will not be overwritten. \
                Please provide an empty directory, or use this dataset")
    

def load_PPB_metadata(location='dataset/extra_experiment/test/PPB-2017-metadata.csv'):
    """
        This function loads the metadata for the PPB dataset
    """
    metadata = []
    with open(location, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            metadata.append(row)
    


