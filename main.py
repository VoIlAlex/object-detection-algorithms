from src import *
from data.references import DATASETS
import src.config as cfg
import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms


# name for model file and log file
MODEL_NAME = 'CIFAR100_MobileNetV2_500_32_SGD_CrossEntropyLoss'
MODEL_FILE_PATH = os.path.join(
    cfg.MY_PRETRAINED_MODELS_PATH,
    MODEL_NAME + '.pth'
)
MODEL_TRAIN_OUTPUT_PATH = os.path.join(
    cfg.MY_PRETRAINED_MODELS_PATH,
    MODEL_NAME + '.log'
)
MODEL_TRAIN_ERRORS_PATH = os.path.join(
    cfg.MY_PRETRAINED_MODELS_PATH,
    MODEL_NAME + '_errors.log'
)


if __name__ == "__main__":
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

    # redirect outputs to
    # the log files
    sys.stdout = open(MODEL_TRAIN_OUTPUT_PATH, 'w+')
    sys.stderr = open(MODEL_TRAIN_ERRORS_PATH, 'w+')

    # save the model meta-info
    print(net)
    print(optimizer)
    print(criterion)

    for epoch in range(500):
        running_loss = 0.0
        for i, (X, y) in enumerate(trainloader):

            # move data to the GPU
            X = X.cuda()
            y = y.cuda()

            # zero the parameter grads
            optimizer.zero_grad()

            # predict
            y_pred = net(X).cuda()

            # backward the loss
            loss = criterion(y_pred, y)
            loss.backward()

            # optimize paramters
            optimizer.step()

            running_loss += loss
            if i % 99 == 0:
                print('[{:6} {:6}] - loss: {}'.format(
                    epoch + 1,
                    i + 1,
                    running_loss
                ))
                sys.stdout.flush()
                running_loss = 0.0

        # save the model
        torch.save(
            obj=net.state_dict(),
            f=MODEL_FILE_PATH
        )
