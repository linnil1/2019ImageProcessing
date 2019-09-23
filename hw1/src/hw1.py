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
import utils


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


if __name__ == "__main__":
    parser = utils.setParser()
    args = parser.parse_args()
    print(args)

    img = utils.read64(args.path)
    new_img = None
    if args.add is not None:
        new_img = imageAdd(img, args.add)
    elif args.addimage is not  None:
        img2 = utils.read64(args.addimage)
        new_img = imageAvg(img, img2)
    elif args.multiply is not None:
        new_img = imageMult(img, args.multiply)
    elif args.special:
        new_img = image_special_func(img)

    # plot
    utils.plot(img, new_img)
