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


# TODO: use kwargs from Research as SUPER kwargs. So they have more priority with regard of items' kwargs

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
                 criterion,
                 watch_test=None):
        Research.__init__(self, research_path, research_scheme)
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._use_cuda = next(model.parameters()).is_cuda
        self._watch_test = watch_test

        # it's used by research items
        if watch_test is not None:
            for item in self._research_scheme:
                item._kwargs['watch_test'] = watch_test

        # save initial model
        internal_path = os.path.join(self._research_path, 'model_init.pth')
        torch.save(self._model.state_dict(), internal_path)

    def start_research_session(self, train_dataloader, test_dataloader=None, epochs=1, iteration_modulo=None):
        """

        Arguments:
            train_dataloader {} -- 

        Keyword Arguments:
            test_dataloader {} --  (default: {None})
            epochs {} --  (default: {1})
            iteration_modulo {} -- per give count of iterations research items will be called
        """

        # update iteration modulo if needed
        if iteration_modulo is not None:
            for item in self._research_scheme:
                item._kwargs['iteration_modulo'] = 1
        else:
            iteration_modulo = 1

        # clear report files
        for item in self._research_scheme:
            internal_path = item._kwargs.get('internal_path', 'report.txt')
            report_path = os.path.join(self._research_path, internal_path)
            if os.path.exists(report_path):
                open(report_path, 'w').close()

        # This dict will be
        # sent to the research
        # items in the research scheme
        network_output = {
            'iteration': None,
            'epoch': None,
            'accuracy_train': None,
            'accuracy_test': None,
            'loss_train': None,
            'loss_test': None,
            'model': self._model,
            'optimizer': self._optimizer,
            'criterion': self._criterion,
            'input': None,
            'output': None,
            'expected_output': None,
            'iterations_per_epoch': min(len(train_dataloader), len(test_dataloader))
        }

        # TODO: here is too many `if self._watch_test:` conditions

        # If only train is needed
        if self._watch_test:
            dataloaders = zip(train_dataloader, test_dataloader)
        else:
            dataloaders = train_dataloader
            network_output['iterations_per_epoch'] = len(train_dataloader)

        # Main cycle
        running_loss_train = 0.0
        if self._watch_test:
            running_loss_test = 0.0
        # TODO: test dataloader has fewer number of samples than train dataloader

        for epoch in range(epochs):
            for i, data in enumerate(dataloaders):

                if self._watch_test:
                    ((X_train, y_train), (X_test, y_test)) = data
                else:
                    (X_train, y_train) = data

                network_output['iteration'] = i
                network_output['epoch'] = epoch
                network_output['input'] = X_train
                network_output['expected_output'] = y_train
                # if using cuda
                if self._use_cuda:
                    X_train = X_train.cuda()
                    y_train = y_train.cuda()
                    if self._watch_test:
                        X_test = X_test.cuda()
                        y_test = y_test.cuda()

                # zero gradients of the parameters
                self._optimizer.zero_grad()

                # calculate the loss
                y_pred = self._model(X_train)
                if self._watch_test:
                    y_pred_test = self._model(X_test)

                if self._use_cuda:
                    y_pred = y_pred.cuda()
                    if self._watch_test:
                        y_pred_test = y_pred_test.cuda()

                network_output['output'] = y_pred

                loss_train = self._criterion(y_pred, y_train)
                if self._watch_test:
                    loss_test = self._criterion(y_pred_test, y_test)

                running_loss_train += loss_train
                if self._watch_test:
                    running_loss_test += loss_test

                # backward propagation
                loss_train.backward()

                # parameters optimization
                self._optimizer.step()

                # absolute iteration
                absolute_iteration = network_output['iterations_per_epoch'] * epoch + i

                if absolute_iteration % iteration_modulo == 0:
                    network_output['loss_train'] = running_loss_train / \
                        iteration_modulo
                    if self._watch_test:
                        network_output['loss_test'] = running_loss_test / \
                            iteration_modulo

                    # write to the report file
                    for item in self._research_scheme:
                        item.print(self._research_path, **network_output)

                    running_loss_train = 0.0

                    if self._watch_test:
                        running_loss_test = 0.0

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

    # items
    from .research_items import ModelConfigurationItem, CurrentIterationItem, LossPrintItem, LossVisualizationItem

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
