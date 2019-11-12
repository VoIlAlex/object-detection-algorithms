"""
    This module contains tools for
    models examination and comparison.

    Using a research scheme you can
    accomplish research of a given
    model in customizable manner.

    Using report scheme you can include
    specific scheme to the report of the
    research.

    * It's kinda network scheme in keras.

"""

import os
import torch
import matplotlib.pyplot as plt


class Research:
    def __init__(self,
                 research_path: str,
                 research_scheme: list):
        self._research_path = research_path
        if not os.path.exists(self._research_path):
            os.makedirs(self._research_path)
        self._research_scheme = research_scheme

    def start_research_session(self):
        raise NotImplementedError(
            "start_research_session(...)"
            " should be implemented."
        )


class ModelResearch(Research):
    def __init__(self,
                 research_path: str,
                 research_scheme: list,
                 model,
                 optimizer,
                 criterion):
        Research.__init__(self, research_path, research_scheme)
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._use_cuda = next(model.parameters()).is_cuda

        # save initial model
        internal_path = os.path.join(self._research_path, 'model_init.pth')
        torch.save(self._model.state_dict(), internal_path)

    def start_research_session(self, train_dataloader, test_dataloader=None, epochs=1):
        # This dict will be
        # sent to the research
        # items in the research scheme

        # clear report files
        for item in self._research_scheme:
            internal_path = item._kwargs.get('internal_path', 'report.txt')
            report_path = os.path.join(self._research_path, internal_path)
            if os.path.exists(report_path):
                open(report_path, 'w').close()

        # output to be passed
        # to the research items
        network_output = {
            'iteration': None,
            'epoch': None,
            'accuracy': None,
            'loss': None,
            'model': self._model,
            'optimizer': self._optimizer,
            'criterion': self._criterion,
            'input': None,
            'output': None,
            'expected_output': None
        }

        # Main cycle
        # TODO: Add test dataloader
        for epoch in range(epochs):
            for i, (X, y) in enumerate(train_dataloader):
                network_output['iteration'] = i
                network_output['epoch'] = epoch
                network_output['input'] = X
                network_output['expected_output'] = y
                # if using cuda
                if self._use_cuda:
                    X = X.cuda()
                    y = y.cuda()

                # zero gradients of the parameters
                self._optimizer.zero_grad()

                # calculate the loss
                y_pred = self._model(X)

                if self._use_cuda:
                    y_pred = y_pred.cuda()

                network_output['output'] = y_pred
                loss = self._criterion(y_pred, y)
                network_output['loss'] = loss

                # backward propagation
                loss.backward()

                # parameters optimization
                self._optimizer.step()

                # write to the report file
                for item in self._research_scheme:
                    item.print(self._research_path, **network_output)

        # save the final model
        model_path = os.path.join(
            self._research_path,
            'model.pth'
        )
        torch.save(self._model.parameters(), model_path)


class ModelsComparison(Research):
    # TODO: Fix everything
    def __init__(self,
                 research_path: str,
                 research_scheme: list,
                 models_dir):
        Research.__init__(self, research_scheme)

        self._models_dir = models_dir

    def start_research_session(self):
        pass


if __name__ == "__main__":
    ### Example ###

    # modules for testing
    import torchvision.models as models
    import torchvision.transforms as transforms
    import torchvision

    # net specific stuff
    net = torchvision.models.MobileNetV2(num_classes=100).cuda()
    optimizer = torch.optim.SGD(params=net.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # load the data
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

    # MAGIC
    research = ModelResearch(
        research_path='trash',  # path to the folder with results
        research_scheme=[  # what to include in report
            ModelConfigurationItem(),
            CurrentIterationItem(),
            LossPrintItem(),
            LossVisualizationItem()
        ],
        model=net,
        optimizer=optimizer,
        criterion=criterion
    )

    research.start_research_session(trainloader, testloader, epochs=10)
