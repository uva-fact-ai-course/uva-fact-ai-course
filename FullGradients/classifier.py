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
from torchvision import datasets, transforms, utils

import os
import argparse
from models.vgg import vgg11
from models.resnet import resnet50
from utils import prepare_data, load_data, CIFAR_100_TRANSFORM_TRAIN, CIFAR_100_TRANSFORM_TEST, CIFAR_10_TRANSFORM


def parse_epoch(dataloader, model, optimizer, criterion, device, train=True):
    '''
        Training and evaluation are put together to avoid duplicated code.
    '''
    if train:
        model.train()
    else:
        model.eval()

    losses, total, correct = 0, 0, 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        # print(data.shape)
        if train:
            optimizer.zero_grad()    
            outputs = model(data)
            loss = criterion.forward(outputs, target)
            loss.backward()
            optimizer.step()

            losses += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # print('batch: %d | Loss: %.3f | Acc: %.3f' % (batch_idx, loss.item(), 100.*predicted.eq(target).sum().item()/target.size(0)))
        else: 
            with torch.no_grad():
                outputs = model(data)
                loss = criterion.forward(outputs, target)
                
                losses += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
    #             print('batch: %d | Loss: %.3f | Acc: %.3f' % (batch_idx, loss.item(), 100.*predicted.eq(target).sum().item()/target.size(0)))
    print("total_batches: %d | total loss: %.3f | epoch Acc: %.3f" % (batch_idx, losses/(batch_idx+1), 100.*correct/total))
    return correct/total

def train(model_config, loader_config):    
    '''
        This function trains the model that is passed in the first argument,
        using the arguments used afterwards.
    '''
    best_acc = 0.0
    for epoch in range(0, model_config.epochs):
        train_acc = parse_epoch(loader_config.trainloader, model_config.model, model_config.optimizer, model_config.criterion, model_config.device)
        torch.cuda.empty_cache()
        model_config.scheduler.step()
        accuracy = parse_epoch(loader_config.testloader, model_config.model, model_config.optimizer, model_config.criterion, model_config.device, train=False)
        
        # if accuracy > best_acc:
        model_config.save_model()
        best_acc = accuracy  

        if train_acc > 0.9:
            break
            
def eval(model, criterion, optimizer, trainloader, testloader, device,
            load_model, save_epochs):
    """
        This function loads the model weights from the load_model location.
        Afterwards it is run through 1 epoch of the test dataset to get the accuracy.
    """

    parse_epoch(testloader, model, optimizer, criterion, device, train=False)