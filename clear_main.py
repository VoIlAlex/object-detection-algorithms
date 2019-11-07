import os
import shutil
from src.config import DEFAULT_MAIN, MAIN_PATH


if __name__ == "__main__":
    while True:
        answer = input('Do you really want to clear main? [y/n] ')
        if answer == 'y':
            with open(MAIN_PATH, 'w+') as f:
                for line in DEFAULT_MAIN:
                    print(line, file=f)
            break
        elif answer == 'n':
            exit(0)
        else:
            print('Unrecognized answer "{}".'.format(answer))
