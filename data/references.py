import json
import os

data_dir = os.path.split(__file__)[0]
references_path = os.path.join(data_dir, 'references.json')
DATASETS = json.load(open(references_path))
