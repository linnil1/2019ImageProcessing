"""
Author: linnil1
Objective: Image Processing HW3
Description: Some operations are implemented in the file
* Min, max, median filter
* laplacian, sobel, rober cross gradient filter
* gaussian ideal butterworth filter
* LoG
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
from functools import wraps


def wrapImgWithPad(func):
    """
    Wrap my own function as a decorator.
    It paded to the image and run on cpp extension then return as np.array
    """
    @wraps(func)
    def wrapFunc(img, size):
        if type(size) is str:
            size = utils.parseSize(size)
        t, l = size // 2
        b, r = size - (t, l) - 1
        data = np.pad(img, ((t, b), (l, r)), "edge")
        return np.array(func(data.tolist(), *img.shape, *size))
    return wrapFunc


@wrapImgWithPad
def medianFilter(*args, **kwargs):
    """ Ordered Filter: Median """
    return utils_cpp.medianFilter(*args, **kwargs)


@wrapImgWithPad
def minFilter(*args, **kwargs):
    """ Ordered Filter: Min """
    return utils_cpp.minFilter(*args, **kwargs)


@wrapImgWithPad
def maxFilter(*args, **kwargs):
    """ Ordered Filter: Max """
    return utils_cpp.maxFilter(*args, **kwargs)


def spatialConv(img, krn):
    """ Convolute the input image and kernel """
    # padding
    size_img = np.array(img.shape)
    size_krn = np.array(krn.shape)
    size = size_img + size_img - 1
    pad_img = np.pad(img, [(0, size[0] - size_img[0]),
                           (0, size[1] - size_img[1])], "constant")
    pad_krn = np.pad(krn, [(0, size[0] - size_krn[0]),
                           (0, size[1] - size_krn[1])], "constant")

    # convolute and inverse
    result_feq = np.fft.fft2(pad_img) * np.fft.fft2(pad_krn)
    result_sp = np.real(np.fft.ifft2(result_feq))[:size_img[0], :size_img[1]]
    return result_sp


def wrapHighboostFilter(func):
    """
    Wrap the highboost coefficient to all high pass filter
    The custom function will return the convolution of img and it's kernel
    """
    @wraps(func)
    @utils.normalizeWrap
    def wrapFunc(img, k=0, *args, needabs=False, **kwagrs):
        res = func(img, *args, **kwagrs)
        if needabs:
            res = np.abs(res)
        return img + k * res
    return wrapFunc


@wrapHighboostFilter
def sobelH(img):
    """ High Pass: Sobel horizontal Highboost(k) """
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return spatialConv(img, kernel)


@wrapHighboostFilter
def sobelV(img):
    """ High Pass: Sobel vertical Highboost(k) """
    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    return spatialConv(img, kernel)


@wrapHighboostFilter
def laplacian4(img):
    """ High Pass: Laplacian4 Highboost(k) """
    kernel = -np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return spatialConv(img, kernel)


@wrapHighboostFilter
def laplacian8(img):
    """ High Pass: Laplacian8 Highboost(k) """
    kernel = -np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    return spatialConv(img, kernel)


@wrapHighboostFilter
def unsharp(img, sig):
    """ High Pass: unsharp with Highboost(k) """
    return img - gaussian(img, sig)


@wrapHighboostFilter
def roberGx(img):
    """ High Pass: Rober Gx with Highboost(k)"""
    kernel = np.array([[-1, 0], [0, 1]])
    return spatialConv(img, kernel)


@wrapHighboostFilter
def roberGy(img):
    """ High Pass: Rober Gy with Highboost(k)"""
    kernel = np.array([[0, -1], [1, 0]])
    return spatialConv(img, kernel)


def customKernal(img, txt):
    """
    Costom kernel convolute with image
    """
    arr = [[i for i in re.split(" |,", t) if i] for t in re.split("\n|;", txt)]
    krn = np.array(arr, dtype=np.float)
    return spatialConv(img, krn)


def boxFilter(img, size):
    """ Order Filter: Mean """
    if type(size) is str:
        size = utils.parseSize(size)
    krn = np.ones(size, dtype=np.float) / size[0] / size[1]
    return spatialConv(img, krn)


def feqOperationXY(img, func):
    """
    Filtering the map by func in frequency domain.
    The input of func is X and Y.
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
    filter_feq = func(i, j)

    # inverse
    filter_img = np.fft.ifft2(filter_feq * feq)
    filter_img[0::2, 1::2] *= -1
    filter_img[1::2, 0::2] *= -1
    result_sp = np.real(filter_img)
    return result_sp


def feqOperation(img, func):
    """
    Filtering the map by func in frequency domain.
    The input of func is a distance to center
    """
    def distfilter(i, j):
        return func(i ** 2 + j ** 2)
    return feqOperationXY(img, distfilter)


def gaussian(img, sig):
    """ Low Pass: Gaussian """
    def gaussianFilter(dist):
        return np.exp(-dist / sig ** 2)
    return feqOperation(img, gaussianFilter)


def idealLowpass(img, cutoff):
    """ Low Pass: Ideal """
    def idealFilter(dist):
        return dist < cutoff ** 2
    return feqOperation(img, idealFilter)


def butterworth(img, cutoff, n):
    """ Low Pass: ButterWorth """
    def butterworthFilter(dist):
        return 1 / (1 + dist / (cutoff ** 2)) ** n
    return feqOperation(img, butterworthFilter)


@wrapHighboostFilter
def gaussianHigh(img, sig):
    """ High Pass: Gaussian with Highboost(k) """
    def gaussianFilter(dist):
        return 1 - np.exp(-dist / sig ** 2)
    return feqOperation(img, gaussianFilter)


@wrapHighboostFilter
def idealHighpass(img, cutoff):
    """ High Pass: Ideal with Highboost(k) """
    def idealFilter(dist):
        return dist > cutoff ** 2
    return feqOperation(img, idealFilter)


@wrapHighboostFilter
def butterworthHigh(img, cutoff, n):
    """ High Pass: ButterWorth with Highboost(k) """
    def butterworthFilter(dist):
        return 1 - 1 / (1 + dist / (cutoff ** 2)) ** n
    return feqOperation(img, butterworthFilter)


@wrapHighboostFilter
def LoG(img, sig):
    """ Laplacian of Gaussian with Highboost(k) """
    n = int(sig * 6)
    n += (n % 2 == 0)
    j, i = np.meshgrid(np.arange(n) - n // 2,
                       np.arange(n) - n // 2)
    krn = (i ** 2 + j ** 2 - 2 * sig ** 2) / (sig ** 4) * \
        np.exp(-(i ** 2 + j ** 2) / (2 * sig ** 2))
    # krn -= krn.sum()
    return spatialConv(img, krn)


def test():
    # read
    real_image = hw2.readRGB("../hw2/data/kemono_friends.jpg")
    gray_image = hw2.toGrayA(real_image)

    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(gaussianHigh(gray_image, 1, 150), cmap="gray")
    plt.show()
    # import timeit
    # print(timeit.timeit(lambda: spatialConv(gray_image, kernel), number=10))
    exit()


def parserAdd_hw3(parser):
    parser.add_argument("--medianfilter",  type=str,   metavar=("aaaxbbb"),
                        func=medianFilter, action=OrderAction)
    parser.add_argument("--minfilter",     type=str,   metavar=("aaaxbbb"),
                        func=minFilter,    action=OrderAction)
    parser.add_argument("--maxfilter",     type=str,   metavar=("aaaxbbb"),
                        func=maxFilter,    action=OrderAction)
    parser.add_argument("--boxfilter",     type=str,   metavar=("123x123"),
                        func=boxFilter,    action=OrderAction)
    parser.add_argument("--ideallowpass",  type=float, metavar=("cutoff"),
                        func=idealLowpass, action=OrderAction)
    parser.add_argument("--gaussian",      type=float, metavar=("cutoff"),
                        func=gaussian,     action=OrderAction)
    parser.add_argument("--butterworth",   type=float, metavar=("cutoff", "n"), nargs=2,
                        func=butterworth,  action=OrderAction)
    parser.add_argument("--idealhighpass", type=float, metavar=("k", "cutoff"), nargs=2,
                        func=idealHighpass,action=OrderAction)
    parser.add_argument("--gaussianhigh",  type=float, metavar=("k", "cutoff"), nargs=2,
                        func=gaussianHigh, action=OrderAction)
    parser.add_argument("--unsharp",       type=float, metavar=("k", "gaussian cutoff"), nargs=2,
                        func=unsharp,      action=OrderAction)
    parser.add_argument("--sobelh",        type=float, metavar=("k"),
                        func=sobelH,       action=OrderAction)
    parser.add_argument("--sobelv",        type=float, metavar=("k"),
                        func=sobelV,       action=OrderAction)
    parser.add_argument("--roberx",        type=float, metavar=("k"),
                        func=roberGx,      action=OrderAction)
    parser.add_argument("--robery",        type=float, metavar=("k"),
                        func=roberGy,      action=OrderAction)
    parser.add_argument("--laplacian4",    type=float, metavar=("k"),
                        func=laplacian4,   action=OrderAction)
    parser.add_argument("--laplacian8",    type=float, metavar=("k"),
                        func=laplacian8,   action=OrderAction)
    parser.add_argument("--kernel",        type=str,   metavar=("arr"),
                        func=customKernal, action=OrderAction)
    parser.add_argument("--log",           type=float, metavar=("k", "sigma"), nargs=2,
                        func=LoG,          action=OrderAction)
    parser.add_argument("--butterworthhigh", type=float, metavar=("k", "cutoff", "n"), nargs=3,
                        func=butterworthHighpass, action=OrderAction)


if __name__ == "__main__":
    # test()
    parser = argparse.ArgumentParser(description="HW3")
    utils.parserAdd_general(parser)
    hw1.parserAdd_hw1(parser)
    hw2.parserAdd_hw2(parser)
    parserAdd_hw3(parser)
    utils.orderRun(parser)
