"""
Author: linnil1
Objective: Image Processing HW1
Description: This program 1)read a spectial format called 64, which represented
a image, 2)do some operation (multiply, add, avg) on it and 3)draw histogram.
"""

import numpy as np
import matplotlib.pyplot as plt
import utils
import argparse
OrderAction = utils.OrderAction


def map64(ch):
    """
    Map 32 character to number
    """
    if ord('0') <= ord(ch) <= ord('9'):
        return ord(ch) - ord('0')
    elif ord('A') <= ord(ch) <= ord('Z'):
        return ord(ch) - ord('A') + 10
    else:
        return None
        raise ValueError(f"Not correct value of {ch}")


def read64(filename):
    """
    Read file that contained 64 formatted image
    """
    read_lines = open(filename).readlines()
    real_image = [[map64(ch) for ch in line if map64(ch) is not None]
                                   for line in read_lines]
    real_image = list(filter(None, real_image))
    real_image = np.array(real_image) / 31
    return real_image


def getHist(img):
    """
    Calculate histogram in image
    """
    arr = np.array([])
    arr.resize(32)
    map32 = np.uint8(img * 31).flatten()
    for i in map32:
        arr[i] += 1
    return arr / arr.sum()


def parserAdd_hw1(parser):
    """
    Add command for operation in hw1
    """
    parser.add_argument('--read64',   type=str,   help="The image you want to read with .64 extension",
                        func=read64,    layer=(0, 1), action=OrderAction)
    parser.add_argument('--add',      type=float, help="Add a number to the image",
                        func=imageAdd,                action=OrderAction)
    parser.add_argument('--diff',     nargs=0,    help="Add a number to the image",
                        func=imageDiff, layer=(2, 0), action=OrderAction)
    parser.add_argument('--avg',      nargs=0,    help="Add a image to the image",
                        func=imageAvg,  layer=(2, 0),      action=OrderAction)
    parser.add_argument('--multiply', type=float, help="Multiply a number to the image",
                        func=imageMult,               action=OrderAction)
    parser.add_argument('--special',  nargs=0,    help="Operator of hw1-2-4",
                        func=image_special_func,      action=OrderAction)


def limitImg(func):
    """
    Limit the image value from 0 to 1.
    Use it as a decorator.
    """
    def wrapFunc(*args, **kwargs):
        img = func(*args, **kwargs)
        img[img > 1] = 1
        img[img < 0] = 0
        return img
    return wrapFunc


@limitImg
def imageMult(img, num):
    """
    Multiple the image by a constant
    """
    return img * num


@limitImg
def imageAdd(img, num):
    """
    Add the image by a constant.
    Make sure the value in my image is from 0 to 1.
    """
    return img + num / 255


def imageDiff(img1, img2):
    """
    Get the difference of two images
    the equation is img1 - img2 and set 0.5 as equal
    """
    if img1.shape != img2.shape:
        raise RuntimeError("Shape is different")
    return (img1 - img2) / 2 + 0.5


@limitImg
def imageAvg(img1, img2):
    """
    Make average of two image pixelwisely.
    """
    if img1.shape != img2.shape:
        raise RuntimeError("Shape is different")
    return (img1 + img2) / 2


@limitImg
def image_special_func(img):
    """
    Special operation to this image
    """
    return img[:, :-1] - img[:, 1:]


def test():
    # read
    # real_image = read64("LINCOLN.64")
    # real_image = read64("JET.64")
    real_image = read64("../LIBERTY.64")
    real_image1 = read64("../LISA.64")

    # now_img = imageAvg(real_image, real_image1)
    now_img = imageAdd(real_image, 100)

    # plot it
    plt.figure()
    plt.imshow(now_img, cmap="gray")
    # histogram
    hist_bin = getHist(now_img)
    print("Separate 0~1 to 32 bins")
    print(hist_bin)
    plt.figure()
    plt.bar(np.arange(32), height=hist_bin)
    # built-in histogram
    # plt.figure()
    # plt.hist(np.int8(real_image.flatten() * 31), bins=32)
    plt.show()


if __name__ == "__main__":
    # Set argument parser
    parser = argparse.ArgumentParser(description="HW1")
    utils.parserAdd_general(parser)
    parserAdd_hw1(parser)
    utils.orderRun(parser)
