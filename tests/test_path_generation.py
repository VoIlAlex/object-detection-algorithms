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

    def test_args_chaining(self):
        class Model():
            pass

        path_generator = ModelPathGenerator('models')
        generated_path = path_generator.generate_path(
            Model(), 'hello', 'world'
        )
        expected_path = os.path.join(
            'models',
            Model().__class__.__name__ + '_hello_world'
        )

        assert generated_path == expected_path

    def test_model_name_from_file_name(self):
        file_name = 'Model1_blahblah'
        model_name = ModelPathGenerator.get_model_name(file_name)
        expected_model_name = 'Model1'
        assert model_name == expected_model_name
