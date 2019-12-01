"""

Note: I need some tools to generate
paths. e.g. to generate paths for 
researches for differnet models.
"""

import os


class PathGenerator:
    def __init__(self, root_path):
        self._root_path = root_path

    def generate_path(self, makedirs: bool = False):
        raise NotImplementedError("PathGenerator cannot be instanciated")


class DefaultPathGenerator(PathGenerator):
    def generate_path(self, internal_path: str, makedirs: bool = False):
        path = os.path.join(
            self._root_path,
            internal_path
        )
        if makedirs and not os.path.exists(path):
            os.makedirs(path)
        return path


class ModelPathGenerator(PathGenerator):
    """Generates path to folder for model 
    log files.
    """

    def generate_path(self,
                      model,
                      optimizer=None,
                      criterion=None,
                      epochs: int = None,
                      dataset_name: str = None,
                      lr: bool = False,
                      batch_size: int = None,
                      makedirs=False, **kwargs):
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

        path = os.path.join(self._root_path, path)

        if makedirs and not os.path.exists(path):
            os.makedirs(path)
        return path


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
