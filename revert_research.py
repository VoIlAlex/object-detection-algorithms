# TODO: script to revert research

import os
import shutil

PATH_TO_MAIN = 'main.py'


if __name__ == "__main__":
    path_to_research = os.path.join('src', '!research')
    while not os.path.isfile(path_to_research):
        internals = os.listdir(path_to_research)

        # choose internal path
        while True:
            for i, internal in enumerate(internals):
                print('{}. {}'.format(i + 1, internal))
            answer = input('Your answer: ')
            try:
                answer = int(answer) - 1
                assert 0 <= answer < len(internals)
                break
            except Exception:
                print('Unrecognized answer! Try again')

        # augment the path
        path_to_research = os.path.join(path_to_research, internals[answer])

    # we got path to our reseach
    # now is time to revert it :)
    while True:
        answer = input(
            'Do you really want to revert your research?\n'
            'Content of main will be erased. [y/n] '
        )
        if answer == 'y':
            os.remove(PATH_TO_MAIN)
            shutil.copy(
                src=path_to_research,
                dst=PATH_TO_MAIN
            )
            break
        elif answer == 'n':
            exit(0)
        else:
            print('Unrecognized answer "{}".'.format(answer))
