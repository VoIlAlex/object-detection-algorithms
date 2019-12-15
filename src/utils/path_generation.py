"""Module for path generation.

Note: I need some tools to generate
paths. e.g. to generate paths for
researches for differnet models.
"""

import os


class PathGenerator:
    """Base class for path all the path generators."""

    def __init__(self, root_path: str):
        """Save path to the root directory.

        Arguments:
            root_path {str} -- path to the root directory

        """
        self._root_path = root_path

    def generate_path(self, makedirs: bool = False):
        """Generate path.

        Keyword Arguments:
            makedirs {bool} -- wether to create directory of the generated path (default: {False})

        Raises:
            NotImplementedError: This method should be implemented by derived classes

        """
        raise NotImplementedError("PathGenerator cannot be instanciated")


class DefaultPathGenerator(PathGenerator):
    """Default path generator."""

    def generate_path(self, internal_path: str, makedirs: bool = False) -> str:
        """Generate path of the folder in the root directory.

        Arguments:
            internal_path {str} -- path to the directory in the root directory

        Keyword Arguments:
            makedirs {bool} -- wether to create directory of the generated path  (default: {False})

        Returns:
            str -- generated path

        """
        path = os.path.join(
            self._root_path,
            internal_path
        )
        if makedirs and not os.path.exists(path):
            os.makedirs(path)
        return path


class ModelPathGenerator(PathGenerator):
    """Generates path to folder for a model."""

    def generate_path(self,
                      model, *args,
                      optimizer=None,
                      criterion=None,
                      epochs: int = None,
                      dataset_name: str = None,
                      lr: bool = False,
                      batch_size: int = None,
                      makedirs=False,
                      session=0,
                      **kwargs) -> str:
        """Generate path for the model report.

        Arguments:
            model {torch.Module} -- model to generate path for

        Keyword Arguments:
            optimizer {} -- optimizer for the network (default: {None})
            criterion {} -- loss function (default: {None})
            epochs {int} -- number of the epochs (default: {None})
            dataset_name {str} -- name of the dataset for training (default: {None})
            lr {bool} -- learning rate (default: {False})
            batch_size {int} -- batch size (default: {None})
            makedirs {bool} -- wether to create directory of the generated path (default: {False})
            session {int} -- session number (default: {0})

        Returns:
            str -- generated path   

        """
        path = model.__class__.__name__
        if optimizer is not None:
            path += '_' + optimizer.__class__.__name__
            if lr is True:
                # part of learning rate after '.'
                path += str(optimizer.defaults['lr']).split('.')[1]
        if criterion is not None:
            path += '_' + criterion.__class__.__name__
        if epochs is not None:
            path += '_' + str(epochs) + 'ep'
        if dataset_name is not None:
            path += '_' + dataset_name
        if batch_size is not None:
            path += '_b' + str(batch_size)

        # Process positional arguments
        for path_item in args:
            path += '_' + str(path_item)

        path = os.path.join(self._root_path, path)

        if makedirs and not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def get_model_name(path: str) -> str:
        """Get name of the model given path.

        Arguments:
            path {str} -- path to extract model name from

        Returns:
            str -- model name

        """
        head, tail = os.path.split(path)
        model_name = tail.split('_')[0]
        return model_name


if __name__ == "__main__":
    import torch
    import torchvision
    import torchvision.transforms as transforms

    net = torchvision.models.MobileNetV2(num_classes=100).cuda()

    optimizer = torch.optim.SGD(
        params=net.parameters(),
        lr=0.001,
        momentum=0.9
    )
    criterion = torch.nn.CrossEntropyLoss()

    gen = ModelPathGenerator('!research')
    path = gen.generate_path(net, optimizer, criterion, 10, 'MNIST')
    print(path)
