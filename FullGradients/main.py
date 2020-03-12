import argparse
from utils import ModelConfiguration, DataLoaderConfiguration, EXTRA_TRANSFORM
import os
from roar_experiment import experiment
from sensitive_transparency import sensitive_transparency
import torch

def main(config):
    torch.manual_seed(config.seed)
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    if config.experiment == 'roar':
        model_config = ModelConfiguration(epochs=80, learning_rate=0.01, device=config.device, checkpoint_path=PATH + 'saved-models/')
        loader_config = DataLoaderConfiguration(path=PATH, data_dir='dataset/')
        experiment(model_config, loader_config)

    elif config.experiment == 'pixel_perturbation':
        model_config = ModelConfiguration()
        loader_config = DataLoaderConfiguration()

        percentages = [0.1, 0.3, 0.5, 0.7, 0.9]
        experiment(model_config, loader_config, percentages)

    elif config.experiment == 'extra':
        model_config = ModelConfiguration(epochs=10, learning_rate=0.1, checkpoint_path=PATH + 'saved-models/', device=config.device, model_name='RESNET-50', experiment='extra', num_classes=2)
        if not os.path.exists('dataset/extra_experiment/'):
            print('dataset does not exist and needs to be downloaded!')
        else:
            loader_config = DataLoaderConfiguration(datasetname='extra_experiment', batch_size=100, path=PATH, transform=EXTRA_TRANSFORM)

            sensitive_transparency(model_config, loader_config)

    else:
        print("experiment does not exist, please select roar, pixel_perturbation or extra")

if __name__== "__main__":
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--experiment', type=str, default='roar', help="Select roar, pixel_perturbation or extra experiment")
    parser.add_argument('--device', type=str, default='cuda:0', help="Select device")
    parser.add_argument('--seed', type=int, default=0, help="Set seed")
    config = parser.parse_args()
    main(config)



        