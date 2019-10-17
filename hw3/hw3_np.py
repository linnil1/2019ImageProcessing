"""
Author: linnil1
Objective: Image Processing HW3
Description: Some operations are implemented in the file
* Min, max, median filter
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import hw1_np as hw1
import hw2_np as hw2
import utils_cpp
import utils
from utils import OrderAction


def parseSize(res):
    """
    Parse size from string.
    The string should be like 123x123
    """
    if not re.match(r"^\s*\d+\s*x\s*\d+\s*$", res):
        raise ValueError("The value is not like this format 123x123")
    return np.array(res.split('x'), dtype=np.int)


def wrapImgWithPad(func):
    """
    Wrap my own function as a decorator.
    It paded to the image and run on cpp extension then return as np.array
    """
    def wrapFunc(img, size):
        if type(size) is str:
            size = parseSize(size)
        t, l = size // 2
        b, r = size - (t, l) - 1
        data = np.pad(img, ((t, b), (l, r)), 'edge')
        return np.array(func(data.tolist(), *img.shape, *size))
    return wrapFunc


@wrapImgWithPad
def medianFilter(*args, **kwargs):
    return utils_cpp.medianFilter(*args, **kwargs)


@wrapImgWithPad
def minFilter(*args, **kwargs):
    return utils_cpp.minFilter(*args, **kwargs)


@wrapImgWithPad
def maxFilter(*args, **kwargs):
    return utils_cpp.maxFilter(*args, **kwargs)


def test():
    # read
    real_image = hw2.readRGB("data/Image 4-1.jpg")
    gray_image = hw2.toGrayA(real_image)
    a = medianFilter(gray_image, np.array([35, 35]))
    import timeit
    # print(timeit.timeit(lambda: minFilter(gray_image, np.array([5, 5])), number=10))
    plt.imshow(a, cmap="gray")
    plt.show()
    exit()


def parserAdd_hw3(parser):
    parser.add_argument('--medianfilter', type=str,  help="Median Filter",
                        func=medianFilter, action=OrderAction)
    parser.add_argument('--minfilter',    type=str,  help="Min Filter",
                        func=minFilter,    action=OrderAction)
    parser.add_argument('--maxfilter',    type=str,  help="Max Filter",
                        func=maxFilter,    action=OrderAction)


if __name__ == "__main__":
    test()
    parser = argparse.ArgumentParser(description="HW2")
    utils.parserAdd_general(parser)
    hw1.parserAdd_hw1(parser)
    hw2.parserAdd_hw2(parser)
    parserAdd_hw3(parser)
    utils.orderRun(parser)
