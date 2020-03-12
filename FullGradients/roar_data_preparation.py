"""
    This file creates adjusted datasets using the saliency map
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils

from classifier import train
# Import saliency methods and models
from models.vgg import *
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from utils import (UNNORMALIZE, create_imagefolder_dir, load_data,
                   replace_pixels, return_k_index_argsort,
                   save_imagefolder_image)


def create_data(percentages, model_cfg, loader_cfg, salience_method="full_grad"):
    # Create train and test dataloader
    train_loader = load_data(1, loader_cfg.transform, False, 1, loader_cfg.data_dir, loader_cfg.dataset, train=True, name=loader_cfg.dataset)
    test_loader = load_data(1, loader_cfg.transform, False, 1, loader_cfg.data_dir, loader_cfg.dataset, train=False, name=loader_cfg.dataset)
    
    # Number of pixels in k percent of the image
    num_pixel_list = [round((percentage * loader_cfg.image_size)) for percentage in percentages]

    # Fetch pretrained model on cifar dataset and create full grad object
    model_cfg.load_model()
    full_grad = FullGrad(model_cfg.model, im_size=(1,3,32,32), device=model_cfg.device)

    # Get adjusted data
    create_salience_based_adjusted_data(train_loader, full_grad, num_pixel_list, percentages, model_cfg.device, salience_method, dataset="train")
    create_salience_based_adjusted_data(test_loader, full_grad, num_pixel_list, percentages, model_cfg.device, salience_method, dataset="test")

def create_data_dirs(percentages, num_classes, salience_method):
    """
        Creates directories to save adjusted images.

        percentages : the percentages of pixels which are adjusted, used to create
                        convenient directory names.
    """
    
    os.mkdir(f'dataset/roar_{salience_method}')

    for percentage in percentages:
        directory = f'dataset/roar_{salience_method}/cifar-{num_classes}-{percentage*100}%-removed' 
        create_imagefolder_dir(directory, num_classes)

def create_adjusted_images_and_save(idx, data, sal_map, target, ks, percentages, num_classes, dataset, method, approach = "zero"):
    """
        Creates adjusted images based on different K's, and saves them.
        
        cam         : Salience map of most important pixels.
        ks          : amount of pixels which are adjusted within the image.
        percentages : the percentages of pixels which are adjusted, used to save images in correct
                      directories.
    """

    

    # Get unnormalized image and pre-process salience map
    image = UNNORMALIZE(data.squeeze())   
    sal_map = sal_map.squeeze().cpu().detach().numpy()
    
    for k, percentage in zip(ks, percentages): 

        # Get k indices and replace within image
        indices = return_k_index_argsort(sal_map, k, method)
        new_image = replace_pixels(image, indices, approach = approach)

        # Save adjusted images
        data_dir = f'dataset/roar_{method}/cifar-{num_classes}-{percentage*100}%-removed'
        save_imagefolder_image(data_dir, target, new_image, idx, dataset)


def create_salience_based_adjusted_data(sample_loader, full_grad, ks, percentages, device, salience_method="full_grad", num_classes=10, dataset="train"):
    """
        Creates adjusted images based on different K's, and saves them.
        
        sample_loader: created dataloader, used to sample the images.
        ks           : amount of pixels which are adjusted within the image.
        percentages  : the percentages of pixels which are adjusted, used to save images in correct
                       directories.
        dataset      : Used to define which set is used.  
    """

    # Creates data directories if needed.
    if not os.path.exists(f'dataset/roar_{salience_method}'):
        create_data_dirs(percentages, num_classes, salience_method)
    else:
        print(f"{dataset}set already created!")

    # Loops over sample loader to creates per sample every adjusted image, and saves them.
    for idx, (data, target) in enumerate(sample_loader):
        data, target = data.to(device).requires_grad_(), target.to(device)

        if salience_method == "full_grad":
            _, salience_map, _ = full_grad.saliency(data)

        elif salience_method == "input_grad":
            salience_map, _, _ = full_grad.saliency(data)

        elif salience_method == "random":
            salience_map = torch.randn(1, 1, 32, 32)
  
        # Find most important pixels, replace and save adjusted image.
        create_adjusted_images_and_save(idx, data, salience_map, target, ks, percentages, num_classes, dataset, salience_method)