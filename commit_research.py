import os
import shutil
import re

# line in main.py
# by default.
DEFAULT_MAIN = [
    'from src.models import *'
]


def extract_research_label(research_file_name: str):
    if not research_file_name.startswith('research_') or \
            not research_file_name.endswith('.py'):
        return 0
    label = research_file_name[len('research_'):][:-len('.py')]
    label = int(label)
    return label


def commit_to_research(research_name: str = None):
    research_dir_path = os.path.join('src', '!research')

    # create default research
    # name if not given
    if research_name == '':
        research_files = os.listdir(research_dir_path)
        max_research_label = 0
        for research_file_name in research_files:
            research_label = extract_research_label(research_file_name)
            max_research_label = max(max_research_label, research_label)
        research_name = 'research_{}.py'.format(max_research_label)

    # commit research
    os.rename('main.py', research_name)
    shutil.move(
        src=research_name,
        dst=os.path.join('src', '!research',)
    )

    # create a new main file
    with open('main.py', 'w+') as main_file:
        for line in DEFAULT_MAIN:
            print(line, file=main_file)


if __name__ == "__main__":
    # Run the script until
    # we get correct input
    while(True):
        answer = input('Do you really want to commit? [y/n]')
        if answer == 'y':
            research_name = input(
                'Enter research name (nothing for default): ')
            commit_to_research(research_name)
            break
        elif answer == 'n':
            exit(0)
        else:
            print('Unrecognized answer "{}".'.format(answer))
