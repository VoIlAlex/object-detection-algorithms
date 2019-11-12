import os
import sys

# make root dir visible
# for importing
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)


import src.utils.model_analysis as analysis

# TODO: test iteration modulo


class TestResearchItems:
    def test_research_item_base_class(self):
        ra = analysis.ResearchItem()

        try:
            ra.print('some_path')
        except NotImplementedError:
            pass

    class TestCurrentIterationItem:
        def test_current_iteration_item_default(self):
            research_path = '.temp'
            cia = analysis.CurrentIterationItem()
            kwargs = {
                'epoch': 10,
                'iteration': 12
            }

            # create research file
            if not os.path.exists(research_path):
                os.makedirs(research_path)

            # clear the report file
            report_path = os.path.join(
                research_path,
                'report.txt'
            )
            open(report_path, 'w').close()

            # print current iteration and epoch
            cia.print(research_path, **kwargs)

            # check the report file
            lines = [line.rstrip() for line in open(
                report_path, 'r').readlines() if line != '']
            assert 'Iteration: [    10,    12]' in lines

            # remove temp file
            os.remove(report_path)

    class TestAccuracyItem:

        def test_accuracy_print_item_default(self):
            research_path = '.temp'
            api = analysis.AccuracyPrintItem()
            kwargs = {
                'accuracy_train': 1,
                'accuracy_test': 0.5
            }

            # create research file
            if not os.path.exists(research_path):
                os.makedirs(research_path)

            # clear the report file
            report_path = os.path.join(
                research_path,
                'report.txt'
            )
            open(report_path, 'w').close()

            # print current iteration and epoch
            api.print(research_path, **kwargs)

            # check the report file
            lines = [line.rstrip() for line in open(
                report_path, 'r').readlines() if line != '']
            assert 'Accuracy: [  1.00,  0.50]' in lines

            # remove temp file
            os.remove(report_path)

    class TestLossPrintItem:

        def test_loss_print_item_default(self):
            research_path = '.temp'
            lpi = analysis.LossPrintItem()
            kwargs = {
                'loss_train': 1,
                'loss_test': 2
            }

            # create research file
            if not os.path.exists(research_path):
                os.makedirs(research_path)

            # clear the report file
            report_path = os.path.join(
                research_path,
                'report.txt'
            )
            open(report_path, 'w').close()

            # print current iteration and epoch
            lpi.print(research_path, **kwargs)

            # check the report file
            lines = [line.rstrip() for line in open(
                report_path, 'r').readlines() if line != '']
            assert 'Loss: [  1.00,  2.00]' in lines

            # remove temp file
            os.remove(report_path)
