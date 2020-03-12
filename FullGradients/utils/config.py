import os

import torch
import torch.nn as nn
import torch.optim as optim

from utils import CIFAR_10_TRANSFORM, load_data, UNNORMALIZE
from models.resnet import resnet50
from models.vgg import vgg11

class ModelConfiguration:
    def __init__(self, num_classes=10, epochs=200, save_epochs=1, learning_rate=0.1, device='cuda:0', class_size=512, model_name='VGG-11', checkpoint_path='saved-models/', experiment='roar'):
        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.model_name = model_name
        self.class_size = class_size
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        
        self.epochs = epochs
        self.save_epochs = save_epochs
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.criterion = nn.CrossEntropyLoss()
        self.model_dir = checkpoint_path + model_name + '-' + experiment + '.pth'
        self.set_model(model_name)
        self.load_model()
        self.set_optimizer()

    def set_model(self, name):
        if name == 'VGG-11':
            self.model = vgg11(pretrained=False, num_classes=self.num_classes, class_size=self.class_size, im_size=(1,3,32,32), device=self.device).to(self.device)

        if name == 'RESNET-50':
            self.model = resnet50(pretrained=False, num_classes=self.num_classes, device=self.device).to(self.device)

    def set_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, nesterov=True)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 120, 160, 200], gamma=0.2)

    def load_model(self):
        if os.path.exists(self.model_dir):
            self.model.load_state_dict(torch.load(self.model_dir), True if self.device == 'cuda:0' else False)

    def save_model(self):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        torch.save(self.model.state_dict(), self.model_dir)
         


class DataLoaderConfiguration:
    def __init__(self, dataset='cifar10', batch_size=128, path='./', transform=CIFAR_10_TRANSFORM, datasetname='cifar-10', data_dir='dataset'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_dir = path + data_dir
        self.num_workers = 2
        self.dataset_name = datasetname
        self.dataset = dataset
        self.path = path

        self.save_path = 'results/'
        self.transform = transform
        self.unnormalize = UNNORMALIZE

        self.trainloader = load_data(batch_size, transform,
                                True, 2, self.data_dir, self.dataset_name, train=True, name=self.dataset)
                                
        self.testloader = load_data(self.batch_size, transform,
                                False, 2, self.data_dir, self.dataset_name, train=False, name=self.dataset)

        sample_img = next(iter(self.trainloader))[0]
        self.image_size = sample_img.shape[2] * sample_img.shape[3]
        self.image_shape = sample_img.shape
