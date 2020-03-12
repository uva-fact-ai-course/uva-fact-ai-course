#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#
""" 
This document contains the functions to classify the images,
with a certain dataLoader. Calling this file as main function
it allows for certain flags to be set and instantly run the classifier
and save it
"""
import torch
from torchvision import datasets, utils
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from models.vgg import vgg11, vgg16, vgg19
from models.resnet import resnet50
from utils import prepare_data, load_data, CIFAR_100_TRANSFORM_TRAIN, CIFAR_100_TRANSFORM_TEST, CIFAR_10_TRANSFORM, load_imageFolder_data
from classifier import train, eval
from saliency.simple_fullgrad import SimpleFullGrad
import csv
from misc_functions import save_saliency_map

def etnic_acc(dataloader, model, optimizer, criterion, device, csv_dir, train=True):
    '''
        Training and evaluation are put together to avoid duplicated code.
    '''
    model.eval()
    metadata = []
    with open(csv_dir, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            metadata.append(row)

    total = {'Male': {'lighter' : 0, 'darker': 0}, 'Female':  {'lighter' : 0, 'darker': 0} }
    correct = {'Male': {'lighter' : 0, 'darker': 0}, 'Female':  {'lighter' : 0, 'darker': 0} }

    image_index = 1
    for batch_idx, (data, target) in enumerate(dataloader):
        # print(batch_idx)
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            outputs = model(data)
            loss = criterion.forward(outputs, target)
            _, predicted = outputs.max(1)

            for i in range(target.size(0)):
                gender = metadata[image_index][2]
                tint = metadata[image_index][4]
                total[gender][tint] += 1
                correct[gender][tint] += predicted[i].eq(target[i]).sum().item()

                image_index += 1

    
    
    print("Male, lighter, accuracy :", correct['Male']['lighter']/total['Male']['lighter'])
    print("Male, darker, accuracy :", correct['Male']['darker']/total['Male']['darker'])
    print("Female, lighter, accuracy :", correct['Female']['lighter']/total['Female']['lighter'])
    print("Female, darker, accuracy :", correct['Female']['darker']/total['Female']['darker'])

def compute_save_fullgrad_saliency(sample_loader, unnormalize, save_path, device, simple_fullgrad):
    for batch_idx, (data, target) in enumerate(sample_loader):
        data, target = data.to(device).requires_grad_(), target.to(device)

        # Compute saliency maps for the input data
        # _, cam, _ = fullgrad.saliency(data)
        cam_simple, _ = simple_fullgrad.saliency(data)
        # Save saliency maps
        for i in range(data.size(0)):
            filename = save_path + str( (batch_idx+1) * (i+1))
            # print(i)
            image = unnormalize(data[i,:,:,:].cpu())
            save_saliency_map(image, cam_simple[i,:,:,:], filename + '.jpg')
            # save_saliency_map(image, cam[i,:,:,:], filename + 'full.jpg')

def sensitive_transparency(model_config, data_config):
    saliency_dir = data_config.path + 'dataset/saliency/'
    dataset = datasets.ImageFolder(root=saliency_dir, transform=data_config.transform)
    saliencyloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)
    # model_config.set_model()
    if not os.path.exists(model_config.model_dir):
        train(model_config, data_config)
    else:    
        model_config.load_model()

    simple_fullgrad = SimpleFullGrad(model_config.model)
    # model
    # fullgrad = FullGrad(model_config.model, im_size=(1,3,32,32), device=model_config.device)

    if os.path.exists(saliency_dir):
        compute_save_fullgrad_saliency(saliencyloader, data_config.unnormalize, data_config.save_path, model_config.device, simple_fullgrad)

    else:
        print("Add pictures to: " + saliency_dir)
        print("Saliency maps will be shown")
        
    csv_dir = data_config.data_dir + '/' + data_config.dataset_name + '/test/PPB-2017-metadata.csv'

    etnic_acc(data_config.testloader, model_config.model, model_config.optimizer, model_config.criterion, model_config.device, csv_dir)
