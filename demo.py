from tkinter import Tk, filedialog
import src.config as cfg

import cv2
import torch

# Config
MODEL_PATH = None
IMAGE_PATH = None


def get_default_model_path() -> str:
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        initialdir=cfg.PRETRAINED_MODELS_PATH,
        title='Select model to demonstrate',
        filetypes=(('PyTorch models', '*.pth'),))

    root.destroy()
    return file_path


def get_default_image_path() -> str:
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        initialdir=cfg.PRETRAINED_MODELS_PATH,
        title='Select model to demonstrate',
        filetypes=(('JPED', '*.jpeg'),))

    root.destroy()
    return file_path


def demo():
    model_path = MODEL_PATH if MODEL_PATH is not None else get_default_model_path()
    image_path = IMAGE_PATH if IMAGE_PATH is not None else get_default_image_path()

    img = cv2.imread(image_path)
    model_dict = torch.load(model_path)


if __name__ == "__main__":
    demo()
