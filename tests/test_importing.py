import os
import sys

# make root dir visible
# for importing
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import src


class TestImporting:
    def test_rcnn_family_importing(self):
        assert hasattr(src, 'RCNN')
        assert hasattr(src, 'FastRCNN')
        assert hasattr(src, 'FasterRCNN')
