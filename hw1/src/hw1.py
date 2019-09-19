"""
Author: linnil1
Objective: Image Processing HW1
Description: This program 1)read a spectial format called 64, which represented 
a image, 2)do some operation (multiply, add, avg) on it and 3)draw histogram.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import myimg


def map64(ch):
    """
    Map 32 character to number
    """
    if ord('0') <= ord(ch) <= ord('9'):
        return ord(ch) - ord('0')
    elif ord('A') <= ord(ch) <= ord('Z'):
        return ord(ch) - ord('A') + 10
    else:
        raise ValueError(f"Not correct value of {ch}")


def read64(filename):
    """
    Read file that contained 64 formatted image
    """
    read_lines = open(filename).readlines()
    real_image = [[map64(ch) for ch in line[:-1]] for line in read_lines]
    real_image = np.array(real_image[:-1]) / 31
    return real_image


def wrapImg(func):
    """
    Wrap my own function return as np.array
    Use it as a decorator.
    """
    def wrapFunc(*args, **kwargs):
        return np.array(func(*args, **kwargs))
    return wrapFunc


@wrapImg
def imageMult(img, num):
    """
    Multiple the image by a constant
    """
    return myimg.imageMult(img.tolist(), num)


@wrapImg
def imageAdd(img, num):
    """
    Add the image by a constant. Make sure the value in my image is from 0 to 1.
    """
    return myimg.imageAdd(img.tolist(), num)


@wrapImg
def imageAvg(img1, img2):
    """
    Make average of two image pixelwisely.
    """
    return myimg.imageAvg(img1.tolist(), img2.tolist())


@wrapImg
def image_special_func(img):
    """
    Special operation to this image
    """
    return myimg.image_special_func(img.tolist())


# read
# real_image = read64("../LINCOLN.64")
# real_image = read64("../JET.64")
real_image = read64("../LIBERTY.64")
real_image1 = read64("../LISA.64")

# plot it
plt.figure()
# plt.imshow(imageAvg(real_image, real_image1), cmap="gray")
plt.imshow(image_special_func(real_image), cmap="gray")

# histogram
# plt.figure()
# plt.hist(np.int8(real_image.flatten() * 31), bins=32)
plt.show()
