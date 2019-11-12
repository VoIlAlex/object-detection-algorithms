import os
from .utils import Visualizer


class ResearchItem:
    """Base class for all the research
    items.
    """

    def print(self, research_path, *args, **kwargs):
        raise NotImplementedError(
            'print(...) method should be implemented '
            'by all the childs of ResearchItem class'
        )


class CurrentIterationItem(ResearchItem):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def print(self, research_path, *args, **kwargs):

        #
        # extract arguments
        #

        # user provided arguments
        print_end = self._kwargs.get('print_end', '\n')
        internal_path = self._kwargs.get('internal_path', 'report.txt')
        iteration_modulo = self._kwargs.get('iteration_modulo', 1)

        # framework provided arguments
        epoch = kwargs.get('epoch', None)
        iteration = kwargs.get('iteration', None)

        # get out if necessary
        if iteration % iteration_modulo != 0:
            return

        # path to file with the report
        file_path = os.path.join(
            research_path,
            internal_path
        )

        # write item to the file
        with open(file_path, 'a') as f:
            print('[{:5d}{:5d}] '.format(
                epoch, iteration), file=f, end=print_end)


class AccuracyItem(ResearchItem):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def print(self, research_path, *args, **kwargs):

        #
        # extract arguments
        #

        # user provided arguments
        print_end = self._kwargs.get('print_end', '\n')
        internal_path = self._kwargs.get('internal_path', 'report.txt')
        iteration_modulo = self._kwargs.get('iteration_modulo', 1)

        # framework provided arguments
        iteration = kwargs.get('iteration', None)
        accuracy = kwargs.get('accuracy', None)

        # get out if necessary
        if iteration % iteration_modulo != 0:
            return

        # path to file with the report
        file_path = os.path.join(
            research_path,
            internal_path
        )

        # write item to the file
        with open(file_path, 'a') as f:
            print('Accuracy: {:.5f} '.format(accuracy),
                  file=f,
                  end=print_end)


class LossPrintItem(ResearchItem):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def print(self, research_path, *args, **kwargs):

        #
        # extract arguments
        #

        # user provided arguments
        print_end = self._kwargs.get('print_end', '\n')
        internal_path = self._kwargs.get('internal_path', 'report.txt')
        iteration_modulo = self._kwargs.get('iteration_modulo', 1)

        # framework provided arguments
        iteration = kwargs.get('iteration', None)
        loss = kwargs.get('loss_train', None)

        # get out if necessary
        if iteration % iteration_modulo != 0:
            return

        # path to file with the report
        file_path = os.path.join(
            research_path,
            internal_path
        )

        # write item to the file
        with open(file_path, 'a') as f:
            print('Loss: {:.5f} '.format(loss),
                  file=f,
                  end=print_end)


class LossVisualizationItem(ResearchItem):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        func_descriptors = [
            {
                'name': 'train',
                'color': 'r'
            },
            {
                'name': 'test',
                'color': 'b'
            }
        ]
        self._visualizer = Visualizer(func_descriptors, 'Iterations', 'Loss')

    def print(self, research_path, *args, **kwargs):

        # user provided arguments
        print_end = self._kwargs.get('print_end', '\n')
        internal_path = self._kwargs.get('internal_path', 'report.txt')
        iteration_modulo = self._kwargs.get('iteration_modulo', 1)

        # framework provided arguments
        iteration = kwargs.get('iteration', None)
        iterations_per_epoch = kwargs.get('iterations_per_epoch', None)
        epoch = kwargs.get('epoch', None)
        loss_train = kwargs.get('loss_train', None)
        loss_test = kwargs.get('loss_test', None)

        # convert the loss
        # to the right format
        loss_train = loss_train.cpu().detach().numpy()
        loss_test = loss_test.cpu().detach().numpy()

        # absolute iteration
        x_pos = iterations_per_epoch * epoch + iteration

        if iteration % iteration_modulo == 0:
            self._visualizer.update({
                'train': {
                    'x': x_pos,
                    'y': loss_train
                },
                'test': {
                    'x': x_pos,
                    'y': loss_test
                }
            })
        self._visualizer.redraw()


class ModelConfigurationItem(ResearchItem):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def print(self, research_path, *args, **kwargs):
        #
        # extract arguments
        #

        # user provided arguments
        print_end = self._kwargs.get('print_end', '\n')
        internal_path = self._kwargs.get('internal_path', 'report.txt')

        # framework provided arguments
        epoch = kwargs.get('epoch', 0)
        iteration = kwargs.get('iteration', 0)

        # it should be printed only once
        if epoch != 0 or iteration != 0:
            return

        model = kwargs.get('model', None)
        optimizer = kwargs.get('optimizer', None)
        criterion = kwargs.get('criterion', None)

        # path to file with the report
        file_path = os.path.join(
            research_path,
            internal_path
        )

        # write item to the file
        with open(file_path, 'a') as f:
            print(model, file=f)
            print(optimizer, file=f)
            print(criterion, file=f, end=print_end)
