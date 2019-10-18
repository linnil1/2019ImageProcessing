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


def spatialConv(img, krn):
    # padding
    size_img = np.array(img.shape)
    size_krn = np.array(krn.shape)
    size = np.max(np.stack([size_img, size_krn]), axis=0)
    pad_img = np.pad(img, [(0, size[0] - size_img[0]),
                           (0, size[1] - size_img[1])], 'constant')
    pad_krn = np.pad(krn, [(0, size[0] - size_krn[0]),
                           (0, size[1] - size_krn[1])], 'constant')

    # convolute and inverse
    result_feq = np.fft.fft2(pad_img) * np.fft.fft2(pad_krn)
    result_sp = np.real(np.fft.ifft2(result_feq))[:size_img[0], :size_img[1]]
    return result_sp


def wrapSharpenFilter(func):
    """
    Wrap the unsharpen coefficient to all high pass filter
    The custom function will return the convolution of img and it's kernal
    """
    @hw1.limitImg
    def wrapFunc(img, k=0, needabs=True, *args, **kwagrs):
        global tmp1
        res = func(img, *args, **kwagrs)
        if needabs:
            res = np.abs(res)
        return img + k * res
    return wrapFunc


@wrapSharpenFilter
def sobelH(img):
    kernal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return spatialConv(img, kernal)


@wrapSharpenFilter
def sobelV(img):
    kernal = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    return spatialConv(img, kernal)


@wrapSharpenFilter
def laplacian4(img):
    kernal = -np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return spatialConv(img, kernal)


@wrapSharpenFilter
def laplacian8(img):
    kernal = -np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    return spatialConv(img, kernal)


@wrapSharpenFilter
def unsharpen(img, sig):
    """
    Unsharpen method: high pass = ori - lowpass
    """
    return img - gaussian(img, sig)


@wrapSharpenFilter
def roberA(img):
    kernal = np.array([[-1, 0], [0, 1]])
    return spatialConv(img, kernal)


@wrapSharpenFilter
def roberB(img):
    kernal = np.array([[0, -1], [1, 0]])
    return spatialConv(img, kernal)


def feqOperation(img, func):
    """
    Editing the map by func in frequency domain.
    The input of func is a distance to center
    """
    # fft
    trans_img = img.copy()
    trans_img[0::2, 1::2] *= -1
    trans_img[1::2, 0::2] *= -1
    feq = np.fft.fft2(trans_img)

    # distance and func
    s = img.shape
    j, i = np.meshgrid(np.arange(s[1]) - s[1] // 2,
                       np.arange(s[0]) - s[0] // 2)
    filter_feq = func(j ** 2 + i ** 2)

    # inverse
    filter_img = np.fft.ifft2(filter_feq * feq)
    filter_img[0::2, 1::2] *= -1
    filter_img[1::2, 0::2] *= -1
    result_sp = np.real(filter_img)
    return result_sp


def gaussian(img, sig):
    """
    Gaussian Blur
    """
    def gaussianFilter(dest):
        return np.exp(-dest / sig ** 2)
    return feqOperation(img, gaussianFilter)


def idealLow(img, cutoff):
    """
    Low pass filetr by ideal cutoff
    """
    def idealFilter(dest):
        return dest < cutoff ** 2
    return feqOperation(img, idealFilter)


def butterWorth(img, cutoff, n):
    """
    ButterWorth low pass filter
    """
    def butterWorthFilter(dest):
        return 1 / (1 + dest / (cutoff ** 2)) ** n
    return feqOperation(img, butterWorthFilter)


def test():
    # read
    real_image = hw2.readRGB("data/Image 3-2.JPG")
    gray_image = hw2.toGrayA(real_image)
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.title("Original")
    plt.imshow(gray_image, cmap="gray")
    plt.subplot(2, 2, 2)
    plt.title("Rober A")
    a = roberA(gray_image, k=2, needabs=True)
    plt.imshow(tmp, cmap="gray")
    plt.subplot(2, 2, 3)
    plt.title("Apply the filter")
    plt.imshow(tmp1, cmap="gray")
    plt.subplot(2, 2, 4)
    plt.title("Apply the filter and add to original")
    plt.imshow(a, cmap="gray")
    plt.show()
    # import timeit
    # print(timeit.timeit(lambda: spatialConv(gray_image, kernal), number=10))
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
