import numpy as np
from PIL import Image


def resize(img, new_width=600, new_height=450):
    return np.array(Image.fromarray(img).resize(size=(new_width, new_height)))

def rotate(img, degrees):
    raise Exception('Not implemented yet')

def flip(img):
    raise Exception('Not implemented yet')
