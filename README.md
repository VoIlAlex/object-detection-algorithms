# Object detection algorithms

[![codecov](https://codecov.io/gh/VoIlAlex/object-detection-algorithms/branch/master/graph/badge.svg?token=yFH2n7WtKe)](https://codecov.io/gh/VoIlAlex/object-detection-algorithms)
![travis-ci](https://travis-ci.com/VoIlAlex/object-detection-algorithms.svg?token=tasqvqAdVwyCcJuhTntj&branch=master)


This repository contains my coursework research in field of neural networks. **It's only aims studying purposes and isn't supposed to be used in commercial projects**.

## Dependencies

All the dependencies of the project are listed in [requirements.txt](requirements.txt). The project is build with help of **keras** deep learning framework.


## Installation

To install the project run the following command:
```
pip install -r requirement.txt
```

## Usage

#### Crafting table

The main research file is [main.py](main.py). It's kinda working area of the project.
To run the script enter this in the terminal:
```
python main.py
```

#### Saving results

When experiment is completed use
```
python3 commit_research
```
to commit the [main.py](main.py) file to the database of researches. The script will ask you to provide name of the research. You might not provide a research name. In that case it will name the research with default name.

## Testing
The tool use for testing this project is `pytest`.

#### Running tests
To run tests run the following command:
```
pytest
```

## Project structure

Project structure should provide convenience of development process, scalability, separation of development parts.

#### Output of `tree` command

```
. 
├── data 
├── docs 
├── models 
│   ├── my_pretrained 
│   └── pretrained 
├── src 
│   ├── !research
│   ├── RCNN 
│   ├── SSD 
│   ├── utils
│   └── YOLO 
└── tests

```

#### Folders overview
* **data** - contains all the training data for nets.
* **docs** - here you can find documentation of the coursework.
* **models** - pre-trained weights of the models.
  * **my_pretrained** - weights of my models.
  * **pretrained** - third-party weights
* **src** - sources of the project
  * **!research** - folder for all the research scripts saved for demonstration.
  * **RCNN** - "region-based neural network" family.
  * **SSD** - "single shot detection" family.
  * **utils** - utility functions for the research.
  * **YOLO** - "you only look once" family.
* **tests** - testing code for the project.

#### Base for all networks

The base class for all the networks is called `ObjectDetectionNet` from [model_template.py](src/model_template.py). It provides routines common for all the networks and an interface for them.

#### Family folders

Structure of a family folder is as follows:
```
<family_name>
├── __init__.py
├── _<family_member_1>.py
├── _<_family_member_2>.py
└── ...
```

where </br>
`__init__.py` - file collector for all the members of a family.</br>
`<family_name>` - RCNN, SSD or YOLO; </br>
`<family_member>` - particular model of the family.

## References
