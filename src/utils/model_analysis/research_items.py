"""Collection of items to be used ass building blocks for research."""
import os
from .utils import Visualizer
import matplotlib.pyplot as plt


class ResearchItem:
    """Base class for all the research items.

    Each derived class of ResearchItem works like
    a filter. Processing application passes all the
    session information (iteration, accuracy, etc.),
    and each Item gets only stuff relevant to its purpose.
    Then it prints generated report string to the 
    specified report file.
    """

    def print(self, research_path: str, *args, **kwargs):
        """Print result of an item to the specified report file.

        Arguments:
            research_path {str} -- path to the reseach folder.

        Raises:
            NotImplementedError: This method is a base class. 
            This method should be implemented by each component separately.

        """
        raise NotImplementedError(
            'print(...) method should be implemented '
            'by all the children of ResearchItem class'
        )

    def final_action(self, research_path, *args, **kwargs):
        """

        Arguments:
            research_path {[type]} -- [description]
        """
        # TODO: Add final action support for items

        pass


class CurrentIterationItem(ResearchItem):
    """Item to deal with current iteration of processing."""

    def __init__(self, *args, **kwargs):
        """Save args and kwargs for later use."""
        self._args = args
        self._kwargs = kwargs

    def print(self, research_path: str, *args, **kwargs):
        """Print currect iteration to the research file.

        Arguments:
            research_path {str} -- path to research folder with reports.

        """
        # # Extract arguments
        # user provided arguments
        print_end = self._kwargs.get('print_end', '\n')
        internal_path = self._kwargs.get('internal_path', 'report.txt')
        iteration_modulo = self._kwargs.get('iteration_modulo', 1)
        use_absolute_iteration = self._kwargs.get('absolute_iteration', False)
        iterations_per_epoch = kwargs.get('iterations_per_epoch', None)

        # framework provided arguments
        epoch = kwargs.get('epoch', -1)
        iteration = kwargs.get('iteration', -1)

        # absolute iterations count
        absolute_iteration = iterations_per_epoch * epoch + iteration

        # get out if necessary
        if iteration % iteration_modulo != 0:
            return

        # path to file with the report
        file_path = os.path.join(
            research_path,
            internal_path
        )

        # write item to the file
        if use_absolute_iteration:
            with open(file_path, 'a') as f:
                print('Iteration: {:10d}'.format(absolute_iteration),
                      file=f,
                      end=print_end)
        else:
            with open(file_path, 'a') as f:
                print('Iteration: [{:6d},{:6d}]'.format(
                    epoch, iteration),
                    file=f,
                    end=print_end)


class AccuracyPrintItem(ResearchItem):
    """Item to deal with accuracy of a model."""

    def __init__(self, *args, **kwargs):
        """Save args and kwargs for later use."""
        self._args = args
        self._kwargs = kwargs

    def print(self, research_path, *args, **kwargs):
        """Print accuracy to the research file.

        Arguments:
            research_path {str} -- path to research folder with reports.

        """
        # # Extract arguments
        # user provided arguments
        print_end = self._kwargs.get('print_end', '\n')
        internal_path = self._kwargs.get('internal_path', 'report.txt')
        iteration_modulo = self._kwargs.get('iteration_modulo', 1)
        watch_test = self._kwargs.get('watch_test', False)

        # framework provided arguments
        iteration = kwargs.get('iteration', -1)
        accuracy_train = kwargs.get('accuracy_train', -1)
        accuracy_test = kwargs.get('accuracy_test', -1)

        # TODO: use absolute iteration
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
            if watch_test:
                print('Accuracy: [{:6.2f},{:6.2f}] '.format(accuracy_train, accuracy_test),
                      file=f,
                      end=print_end)
            else:
                print('Accuracy: {:6.2f}'.format(accuracy_train),
                      file=f,
                      end=print_end)


class LossPrintItem(ResearchItem):
    """Item to deal with loss of a training/testing."""

    def __init__(self, *args, **kwargs):
        """Save args and kwargs for later use."""
        self._args = args
        self._kwargs = kwargs

    def print(self, research_path, *args, **kwargs):
        """Print loss of training/testing to the research file.

        Arguments:
            research_path {str} -- path to research folder with reports.

        """
        # # Extract arguments
        # user provided arguments
        print_end = self._kwargs.get('print_end', '\n')
        internal_path = self._kwargs.get('internal_path', 'report.txt')
        iteration_modulo = self._kwargs.get('iteration_modulo', 1)
        watch_test = self._kwargs.get('watch_test', False)

        # framework provided arguments
        iteration = kwargs.get('iteration', -1)
        loss_train = kwargs.get('loss_train', None)
        loss_test = kwargs.get('loss_test', None)

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
            if watch_test:
                print('Loss: [{:6.2f},{:6.2f}] '.format(loss_train, loss_test),
                      file=f,
                      end=print_end)
            else:
                print('Loss: {:6.2f}'.format(loss_train),
                      file=f,
                      end=print_end)


class LossVisualizationItem(ResearchItem):
    """Item to deal with visualization of the training/testing loss of a model."""

    def __init__(self, watch_test: bool = False, img_name: str = 'loss_rate.png', *args, **kwargs):
        """Save args and kwargs for later use.

        Arguments:
            watch_test {bool} -- whether to plot testing loss.
            img_name {str} -- name of the image to save plot to.

        """
        self._args = args
        self._kwargs = kwargs
        if watch_test is True:
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
        else:
            func_descriptors = [
                {
                    'name': 'train',
                    'color': 'r'
                }
            ]
        self._watch_test = watch_test
        self._visualizer = Visualizer(func_descriptors, 'Iterations', 'Loss')
        self._research_path = 'loss_rate.png'
        self._img_name = img_name

    def print(self, research_path, *args, **kwargs):
        """Save plot to the research folder and update plot state.

        Arguments:
            research_path {str} -- path to research folder with reports.

        """
        # ? Why do I save research path to self ?
        self._research_path = research_path
        # # Extract arguments
        # user provided arguments
        print_end = self._kwargs.get('print_end', '\n')
        internal_path = self._kwargs.get(
            'internal_path', 'report.txt')
        iteration_modulo = self._kwargs.get('iteration_modulo', 1)

        # framework provided arguments
        iteration = kwargs.get('iteration', -1)
        iterations_per_epoch = kwargs.get('iterations_per_epoch', None)
        epoch = kwargs.get('epoch', -1)
        loss_train = kwargs.get('loss_train', None)
        loss_test = kwargs.get('loss_test', None)

        # convert the loss
        # to the right format
        loss_train = loss_train.cpu().detach().numpy()
        if self._watch_test:
            loss_test = loss_test.cpu().detach().numpy()

        # absolute iteration
        x_pos = iterations_per_epoch * epoch + iteration

        if iteration % iteration_modulo == 0:
            if self._watch_test is True:
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
            else:
                self._visualizer.update({
                    'train': {
                        'x': x_pos,
                        'y': loss_train
                    }
                })
        self._visualizer.redraw()


class ModelConfigurationItem(ResearchItem):
    """Item to deal with configuration of a model."""

    def __init__(self, *args, **kwargs):
        """Save args and kwargs for later use."""
        self._args = args
        self._kwargs = kwargs

    def print(self, research_path, *args, **kwargs):
        """Print accuracy to the research file.

        Arguments:
            research_path {str} -- path to research folder with reports.

        """
        # # Extract arguments
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
