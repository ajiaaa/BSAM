import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import numpy as np
from utility.cutout import Cutout

class Cifar10:
    def __init__(self, batch_size, threads):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])



        train_set = torchvision.datasets.CIFAR10(root='E:/Datasets/CIFAR-10', train=True, download=True, transform=train_transform)
        print('len(train_set):', len(train_set))
        test_set = torchvision.datasets.CIFAR10(root='E:/Datasets/CIFAR-10', train=False, download=True, transform=test_transform)
        print('len(test_set):', len(test_set))

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='E:/Datasets/CIFAR-10', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

class Cifar100:
    def __init__(self, batch_size, threads):

        mean, std = np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0

        train_transform = transforms.Compose([
            transforms.Resize(224),
            torchvision.transforms.RandomCrop(size=(224, 224), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR100(root='E:/Datasets/CIFAR-100', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(root='E:/Datasets/CIFAR-100', train=False, download=True,
                                                 transform=test_transform)

        train_sampler = RandomSampler(train_set)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=threads,
                                                 sampler=train_sampler)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)


