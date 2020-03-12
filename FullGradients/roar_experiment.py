"""
    This file contains all the functions needed to run the roar experiment
"""
import os
from roar_data_preparation import create_data
from classifier import train, parse_epoch
from utils import load_imageFolder_data, DataLoaderConfiguration
from saliency.fullgrad import FullGrad


def experiment(model_config, loader_config, percentages = [0.1, 0.3, 0.5, 0.7, 0.9]):
    if not os.path.exists(model_config.model_dir):
        print("Cifar-10 model will be trained which is used for data preparation.")
        train(model_config, loader_config)
        

    If adjusted data is not created, create it. 
    if not os.path.exists(loader_config.path + 'dataset/roar_full_grad/'):
        print("The data for full grad is not found in dataset/roar_full_grad")
        print("Creating it can take a long time, please abort this run and download it from github")
        create_data(percentages, model_config, loader_config, salience_method="full_grad")

    if not os.path.exists(loader_config.path + 'dataset/roar_input_grad/'):
        print("The data for input grad is not found in dataset/roar_input_grad")
        print("Creating it can take a long time, please abort this run and download it from github")
        create_data(percentages, model_config, loader_config, salience_method="input_grad")

    if not os.path.exists(loader_config.path + 'dataset/roar_random/'):
        print("The data for random is not found in dataset/roar_random")
        print("Creating it can take a long time, please abort this run and download it from github")
        create_data(percentages, model_config, loader_config, salience_method="random")

    # Train model based on certrain adjusted data
    accuracy_list = []
    accuracy_list.append(perform_experiment(percentages, model_config, loader_config, "full_grad"))
    accuracy_list.append(perform_experiment(percentages, model_config, loader_config, "input_grad"))
    accuracy_list.append(perform_experiment(percentages, model_config, loader_config, "random"))
    return accuracy_list

def perform_experiment(percentages, model_config, loader_config, method):
    accuracy_list = []

    for percentage in percentages:
        # print(f"Training of model based on {percentage*100}% deletion of pixels.")

        model_config.set_model('VGG-11')
        model_config.set_optimizer()

        data_dir = f'dataset/roar_{method}/'
        datasetname = f'cifar-{model_config.num_classes}-{percentage*100}%-removed/'

        percentage_loader = DataLoaderConfiguration(path=loader_config.path, data_dir=data_dir, datasetname=datasetname)

        model_config.model_dir = f'saved-models/VGG-11-ROAR-{method}-{percentage*100}.pth'

        if not os.path.exists(model_config.model_dir):
            print(f"Model for {percentage*100}% will be trained now.")
            train(model_config, percentage_loader)
        else:
            model_config.load_model()

        eval_accuracy = parse_epoch(percentage_loader.testloader, model_config.model, model_config.optimizer, model_config.criterion, model_config.device, train=False)
        accuracy_list.append(eval_accuracy)
        
        # print("Eval accur:", eval_accuracy)
        # print("----------------------------------------------")

    return accuracy_list
