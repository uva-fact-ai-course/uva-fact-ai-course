from torchvision import transforms
from misc_functions import NormalizeInverse

CIFAR_10_TRANSFORM = transforms.Compose(
    [transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

EXTRA_TRANSFORM = transforms.Compose(
    [transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

UNNORMALIZE = NormalizeInverse(mean = [0.5, 0.5, 0.5],
                    std = [0.5, 0.5, 0.5])

CIFAR_100_TRANSFORM_TRAIN = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

CIFAR_100_TRANSFORM_TEST = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
