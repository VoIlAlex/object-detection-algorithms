import os
import sys

# make root dir visible
# for importing
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from src.utils.path_generation import ModelPathGenerator


class TestPathGeneration:
    def test_path_generation(self):
        class Model:
            pass

        class Optimizer:
            defaults = {
                'lr': 0.001
            }

        class Criterion:
            pass

        epochs = 100
        dataset_name = 'COCO'
        lr = True
        batch_size = 10
        markdirs = False

        path_generator = ModelPathGenerator('models')
        generated_path = path_generator.generate_path(
            model=Model(),
            optimizer=Optimizer(),
            criterion=Criterion(),
            epochs=epochs,
            dataset_name=dataset_name,
            lr=lr,
            batch_size=batch_size,
            markdirs=markdirs
        )
        exprected_path = os.path.join(
            'models',
            'Model_Optimizer001_Criterion_100ep_COCO_b10'
        )

        assert generated_path == exprected_path

    def test_model_required(self):
        path_generator = ModelPathGenerator('models')
        try:
            generated_path = path_generator.generate_path()
        except TypeError:
            pass
