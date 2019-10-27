import os
import sys

# make root dir visible
# for importing
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from data.references import DATASETS
from src.utils.testutils import *


@allow_MoyaPizama_constrains
class TestReferences:

    @MoyaPizama_specific_method
    def test_coco_train(self):
        assert os.path.exists(DATASETS['coco']['train']['images'])
        assert os.path.exists(DATASETS['coco']['train']['labels'])

    @MoyaPizama_specific_method
    def test_coco_test(self):
        assert os.path.exists(DATASETS['coco']['test']['images'])
        assert os.path.exists(DATASETS['coco']['test']['labels'])
