from src import *
from data.references import DATASETS
import src.config as cfg
import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms

# class for model folder path generation
from src.utils.path_generation import ModelPathGenerator

# Import default research
from src.utils.model_analysis import ModelResearch

# Import research items
from src.utils.model_analysis import LossPrintItem
from src.utils.model_analysis import LossVisualizationItem
from src.utils.model_analysis import CurrentIterationItem
from src.utils.model_analysis import ModelConfigurationItem

EPOCHS_COUNT = 100

if __name__ == "__main__":
    # model, dataset, loss, optimizer
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data/datasets', train=True,
                                             download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./data/datasets', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)

    net = torchvision.models.MobileNetV2(num_classes=100).cuda()

    optimizer = torch.optim.SGD(
        params=net.parameters(),
        lr=0.001,
        momentum=0.9
    )
    criterion = torch.nn.CrossEntropyLoss()

    # generate path
    path_generator = ModelPathGenerator(
        root_path=cfg.MY_PRETRAINED_MODELS_PATH
    )
    path = path_generator.generate_path(
        model=net,
        optimizer=optimizer,
        criterion=criterion,
        epochs=EPOCHS_COUNT,
        dataset_name='CIFAR100',
        makedirs=True
    )

    # research
    research = ModelResearch(
        research_path=path,
        research_scheme=[
            ModelConfigurationItem(),
            CurrentIterationItem(print_end=' ', iteration_modulo=10),
            LossPrintItem(iteration_modulo=10),
            LossVisualizationItem(iteration_modulo=10)
        ],
        model=net,
        optimizer=optimizer,
        criterion=criterion
    )

    research.start_research_session(
        trainloader, testloader, epochs=EPOCHS_COUNT, iteration_modulo=50)
