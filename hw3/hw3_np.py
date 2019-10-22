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


def wrapImgWithPad(func):
    """
    Wrap my own function as a decorator.
    It paded to the image and run on cpp extension then return as np.array
    """
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
    size = size_img + size_img - 1
    pad_img = np.pad(img, [(0, size[0] - size_img[0]),
                           (0, size[1] - size_img[1])], "constant")
    pad_krn = np.pad(krn, [(0, size[0] - size_krn[0]),
                           (0, size[1] - size_krn[1])], "constant")

    # convolute and inverse
    result_feq = np.fft.fft2(pad_img) * np.fft.fft2(pad_krn)
    result_sp = np.real(np.fft.ifft2(result_feq))[:size_img[0], :size_img[1]]
    return result_sp


def wrapSharpenFilter(func):
    """
    Wrap the unsharp coefficient to all high pass filter
    The custom function will return the convolution of img and it's kernel
    """
    @hw1.limitImg
    def wrapFunc(img, k=0, *args, needabs=False, **kwagrs):
        res = func(img, *args, **kwagrs)
        if needabs:
            res = np.abs(res)
        return img + k * res
    return wrapFunc


@wrapSharpenFilter
def sobelH(img):
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return spatialConv(img, kernel)


@wrapSharpenFilter
def sobelV(img):
    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    return spatialConv(img, kernel)


@wrapSharpenFilter
def laplacian4(img):
    kernel = -np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return spatialConv(img, kernel)


@wrapSharpenFilter
def laplacian8(img):
    kernel = -np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    return spatialConv(img, kernel)


@wrapSharpenFilter
def unsharp(img, sig):
    """
    Unsharp method: high pass = ori - lowpass
    """
    return img - gaussian(img, sig)


@wrapSharpenFilter
def roberA(img):
    kernel = np.array([[-1, 0], [0, 1]])
    return spatialConv(img, kernel)


@wrapSharpenFilter
def roberB(img):
    kernel = np.array([[0, -1], [1, 0]])
    return spatialConv(img, kernel)


def customKernal(img, txt):
    """
    Transfer a text to kernel array
    """
    arr = [[i for i in re.split(" |,", t) if i] for t in re.split("\n|;", txt)]
    krn = np.array(arr, dtype=np.float)
    return spatialConv(img, krn)


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


def idealLowpass(img, cutoff):
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


def LoG(img, sig, threshold=0.2):
    """
    Laplacian of Gaussian with threshold
    """
    n = int(sig * 6)
    n += (n % 2 == 0)
    j, i = np.meshgrid(np.arange(n) - n // 2,
                       np.arange(n) - n // 2)
    krn = (i ** 2 + j ** 2 - 2 * sig ** 2) / (sig ** 4) * \
        np.exp(-(i ** 2 + j ** 2) / (2 * sig ** 2))
    # krn -= krn.sum()
    res = spatialConv(img, krn)
    threshold = res.max() * threshold
    return res > threshold


def test():
    # read
    real_image = hw2.readRGB("data/Image 3-2.JPG")
    gray_image = hw2.toGrayA(real_image)

    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.title("laplacian after gaussian")
    plt.imshow(laplacian8(gaussian(gray_image, 100), 1), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("gaussian after laplacian")
    plt.imshow(gaussian(laplacian8(gray_image, 1), 100), cmap="gray")
    plt.show()
    # import timeit
    # print(timeit.timeit(lambda: spatialConv(gray_image, kernel), number=10))
    exit()


def parserAdd_hw3(parser):
    parser.add_argument("--medianfilter",  type=str,   metavar=("aaaxbbb"),
                        func=medianFilter, action=OrderAction,
                        help="Median Filter")
    parser.add_argument("--minfilter",     type=str,   metavar=("aaaxbbb"),
                        func=minFilter,    action=OrderAction,
                        help="Min Filter")
    parser.add_argument("--maxfilter",     type=str,   metavar=("aaaxbbb"),
                        func=maxFilter,    action=OrderAction,
                        help="Max Filter")
    parser.add_argument("--ideallowpass",  type=float, metavar=("cutoff"),
                        func=idealLowpass, action=OrderAction,
                        help="Low pass: ideal(cutoff)")
    parser.add_argument("--gaussian",      type=float, metavar=("cutoff"),
                        func=gaussian,     action=OrderAction,
                        help="Low pass: gaussian(cutoff)")
    parser.add_argument("--butterworth",   type=float, metavar=("cutoff", "n"), nargs=2,
                        func=butterWorth,  action=OrderAction,
                        help="Low pass: butterworth(cutoff, n)")
    parser.add_argument("--unsharp",       type=float, metavar=("k", "gaussin cutoff"), nargs=2,
                        func=unsharp,      action=OrderAction,
                        help="High pass: (ori - gaussian(cutoff)) * k + ori")
    parser.add_argument("--sobelh",        type=float, metavar=("k"),
                        func=sobelH,       action=OrderAction,
                        help="High pass: sobel horizontal * k + ori")
    parser.add_argument("--sobelv",        type=float, metavar=("k"),
                        func=sobelV,       action=OrderAction,
                        help="High pass: sobel vertical * k + ori")
    parser.add_argument("--robera",        type=float, metavar=("k"),
                        func=roberA,       action=OrderAction,
                        help="High pass: Rober cross-gradient * k + ori")
    parser.add_argument("--roberb",        type=float, metavar=("k"),
                        func=roberB,       action=OrderAction,
                        help="High pass: Rober cross-gradient * k + ori")
    parser.add_argument("--laplacian4",    type=float, metavar=("k"),
                        func=laplacian4,   action=OrderAction,
                        help="High pass: Laplacian_4neightbor * k + ori")
    parser.add_argument("--laplacian8",    type=float, metavar=("k"),
                        func=laplacian8,   action=OrderAction,
                        help="High pass: Laplacian_8neightbor * k + ori")
    parser.add_argument("--kernel",        type=str, metavar=("arr"),
                        func=customKernal, action=OrderAction,
                        help="Convolute with your custom array: e.g.  \"0 0 0;0 1 0; 0 0 0\" ")
    parser.add_argument("--log",           type=float, metavar=("sigma", "threshold"), nargs=2,
                        func=LoG,          action=OrderAction,
                        help="Laplacian of Gaussian(Threshold = max * threshold)")


if __name__ == "__main__":
    # test()
    parser = argparse.ArgumentParser(description="HW3")
    utils.parserAdd_general(parser)
    hw1.parserAdd_hw1(parser)
    hw2.parserAdd_hw2(parser)
    parserAdd_hw3(parser)
    utils.orderRun(parser)
