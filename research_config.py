import src.utils.model_analysis as analysis

import torch
import torchvision
import torchvision.transforms as transforms

# class for model folder path generation
from src.utils.path_generation import ModelPathGenerator

# Import default research
from src.utils.model_analysis import ModelResearch
# Import research items
from src.utils.model_analysis import *

# Define model specific stuff
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])
trainset = torchvision.datasets.CIFAR100(root='./data/datasets', train=True,
                                         download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR100(root='./data/datasets', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=2)
net = torchvision.models.DenseNet(num_classes=100).cuda()

# optimizers list
optimizer_001 = torch.optim.SGD(
    params=net.parameters(),
    lr=0.001,
    momentum=0.9
)
optimizer_01 = torch.optim.SGD(
    params=net.parameters(),
    lr=0.01,
    momentum=0.9
)
optimizer_1 = torch.optim.SGD(
    params=net.parameters(),
    lr=0.1,
    momentum=0.9
)

criterion = torch.nn.CrossEntropyLoss()


# Configuration of the research session
RESEARCH_SESSION_CONFIG = [
    {
        'construction': {
            'model': net,
            'optimizer': optimizer_001,
            'criterion': criterion,
            'watch_test': False,
            'research_scheme': [
                ModelConfigurationItem(),
                CurrentIterationItem(print_end=' ', iteration_modulo=10),
                LossPrintItem(iteration_modulo=10),
                LossVisualizationItem(iteration_modulo=10)
            ]
        },
        'session': {
            'train_dataloader': trainloader,
            'test_dataloader': testloader,
            'dataset_name': 'CIFAR100',
            'epochs': 100,
            'iteration_modulo': 30,
            'batch_size': 16
        }
    },
    {
        'construction': {
            'model': net,
            'optimizer': optimizer_01,
            'criterion': criterion,
            'watch_test': False,
            'research_scheme': [
                ModelConfigurationItem(),
                CurrentIterationItem(print_end=' ', iteration_modulo=10),
                LossPrintItem(iteration_modulo=10),
                LossVisualizationItem(iteration_modulo=10)
            ]
        },
        'session': {
            'train_dataloader': trainloader,
            'test_dataloader': testloader,
            'dataset_name': 'CIFAR100',
            'epochs': 100,
            'iteration_modulo': 30,
            'batch_size': 16
        }
    },
    {
        'construction': {
            'model': net,
            'optimizer': optimizer_1,
            'criterion': criterion,
            'watch_test': False,
            'research_scheme': [
                ModelConfigurationItem(),
                CurrentIterationItem(print_end=' ', iteration_modulo=10),
                LossPrintItem(iteration_modulo=10),
                LossVisualizationItem(iteration_modulo=10)
            ]
        },
        'session': {
            'train_dataloader': trainloader,
            'test_dataloader': testloader,
            'dataset_name': 'CIFAR100',
            'epochs': 100,
            'iteration_modulo': 30,
            'batch_size': 16
        }
    }
]
