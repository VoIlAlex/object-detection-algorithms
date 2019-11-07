from src import *
from data.data_loading import CocoStyleDataGenerator
from data.data_loading import ClassificationLabelParser
from data.references import DATASETS

import torch
import torch.nn as nn
from torch.functional import F
from torch.optim import SGD


class SomeModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(3, 5, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(5, 10, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(10, 15, kernel_size=(8, 8), stride=(2, 2))
        self.conv4 = nn.Conv2d(15, 20, kernel_size=(5, 5))
        self.fc1 = nn.Linear(213 * 213 * 20, 80)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = F.relu(self.conv4(X))
        X = X.view(-1, 213 * 213 * 20)
        return F.softmax(self.fc1(X))


if __name__ == "__main__":
    train_generator = CocoStyleDataGenerator(
        images_dir=DATASETS['coco']['train']['images'],
        labels_dir=DATASETS['coco']['train']['labels'],
        names_path=DATASETS['coco']['names'],
        label_parser=ClassificationLabelParser()
    )
    test_generator = CocoStyleDataGenerator(
        images_dir=DATASETS['coco']['test']['images'],
        labels_dir=DATASETS['coco']['test']['labels'],
        names_path=DATASETS['coco']['names'],
        label_parser=ClassificationLabelParser()
    )

    net = SomeModel()
    net.to('cuda:0')
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = SGD(params=net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_generator):
            X, y = data
            X = torch.from_numpy(X).permute(0, 3, 1, 2)
            y = torch.from_numpy(y)

            # convert from bytes
            # to float
            X = X.float()
            y = y.long()

            # move to GPU
            X = X.to('cuda:0')
            y = y.to('cuda:0')

            optimizer.zero_grad()

            y_pred = net(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
            if i % 10 == 0:    # print every 2000 mini-batches
                with open('output.txt', 'w') as f:
                    print('[{}, {}] loss: {}'.format
                          (epoch + 1, i + 1, running_loss), file=f)
                    f.flush()
                running_loss = 0.0

            del X
            del y
            del y_pred
            torch.cuda.empty_cache()

        # save model
        PATH = './data/some_model.pth'
        torch.save(net.state_dict(), PATH)
