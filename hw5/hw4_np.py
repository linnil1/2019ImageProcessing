"""
Author: linnil1
Objective: Image Processing HW4
Description: Some operations are implemented in the file
* Homomorphic filter
* gaussian noise
* Wiener filter
* Inverse filter
* Blur: motion and turbulence
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import hw1_np as hw1
import hw2_np as hw2
import hw3_np as hw3
import utils_cpp
import utils
from utils import OrderAction
from functools import wraps


eps = 1e-6


@hw1.limitImg
def homomorphic(img, f_low, f_high, cutoff, c=1):
    """ Homomorphic Filter """
    def homoFilter(dist):
        return f_low + (f_high - f_low) * (1 - np.exp(-c * dist / cutoff ** 2))
    return np.exp(hw3.feqOperation(np.log(img + 1), homoFilter) - 1)


@utils.normalizeWrap
def invFilter(img, formula):
    """ Reconstruction by genernal inverse filter """
    def filterInv(i, j):
        filter_feq = formula(i, j)
        filter_feq[np.abs(filter_feq) < eps] = np.inf
        return filter_feq ** -1
    return hw3.feqOperationXY(img, filterInv)


@utils.normalizeWrap
def wienerFilter(img, k, formula):
    """ Degradation by Motion  """
    def wiener(i, j):
        feq = formula(i, j)
        return feq.conj() / (np.abs(feq) ** 2 + k)
    return hw3.feqOperationXY(img, wiener)


@hw1.limitImg
def noiseGaussian(img, mean, sig):
    """ Add noise: gaussian """
    return img + np.random.normal(mean / 255, sig / 255, img.shape)


@hw1.limitImg
def noiseUniform(img, width):
    """ Add noise: uniform """
    return img + np.random.normal(-width / 2 / 255, width / 2 / 255, img.shape)


@hw1.limitImg
def turbulenceBlur(img, k):
    """ Degradation by turbulence """
    def turbulenceFilter(dist):
        return np.exp(-k * dist ** (5 / 6))
    return hw3.feqOperation(img, turbulenceFilter)


# Four belows functions are motion blur related
def motionFilter(dx, dy):
    """ A motion blur formula """
    def f(i, j):
        uv = (dx * j + dy * i) * np.pi
        filter_feq = np.sin(uv) * np.exp(-1j * uv) / uv
        filter_feq[np.abs(uv) < eps] = 0
        return filter_feq
    return f


@utils.normalizeWrap
def motionBlur(img, dx, dy):
    """ Degradation by Motion  """
    return hw3.feqOperationXY(img, motionFilter(dx, dy))


def motionInv(img, dx, dy):
    """ Reconstruction Motion by Inverse """
    return invFilter(img, motionFilter(dx, dy))


def motionWiener(img, k, dx, dy):
    """ Reconstruction Motion by Wiener """
    return wienerFilter(img, k, motionFilter(dx, dy))


@utils.normalizeWrap
def getShowSpectrum(img):
    """ Spectrum of image """
    if len(img.shape) == 3:
        raise ValueError("The image should be in gray scale")
    img = img.copy()
    img[0::2, 1::2] *= -1
    img[1::2, 0::2] *= -1
    fft_image = np.fft.fft2(img)
    return np.log(np.abs(fft_image) + 1)


def showSpectrum(img):
    """ Spectrum of image """
    plt.figure()
    plt.imshow(getShowSpectrum(img), cmap="gray")


def parserAdd_hw4(parser):
    parser.add_argument("--homomorphic",     type=float, metavar=("f_low", "f_high", "cutoff"), nargs=3,
                        func=homomorphic,    action=OrderAction)
    parser.add_argument("--gaussiannoise",   type=float, metavar=("mean", "std"), nargs=2,
                        func=noiseGaussian,  action=OrderAction)
    parser.add_argument("--uniformnoise",    type=float, metavar=("width"),
                        func=noiseUniform,   action=OrderAction)
    parser.add_argument("--turbulenceblur",  type=float, metavar=("k"),
                        func=turbulenceBlur, action=OrderAction)
    parser.add_argument("--motionblur",      type=float, metavar=("dx", "dy"), nargs=2,
                        func=motionBlur,     action=OrderAction)
    parser.add_argument("--motioninv",       type=float, metavar=("dx", "dy"), nargs=2,
                        func=motionInv,      action=OrderAction)
    parser.add_argument("--motionwiener",    type=float, metavar=("k", "dx", "dy"), nargs=3,
                        func=motionWiener,   action=OrderAction)
    parser.add_argument("--showspectrum",    layer=(1, None), nargs=0,
                        func=showSpectrum,   action=OrderAction)


def test():
    # read
    # real_image = hw2.readRGB("../hw3/data/Image 3-3.jpg")
    # real_image = hw2.readRGB("../hw2/data/kemono_friends.jpg")
    real_image = hw2.readRGB("data/C1HW04_IMG02_2019.bmp")
    gray_image = hw2.toGrayA(real_image)

    std = 20
    # blur_img = motionBlur(gray_image, .1, .1)
    # noise_img = noiseGaussian(blur_img, 0, std / 255)
    noise_img = noiseUniform(gray_image, 100)

    plt.figure(figsize=(6, 12))
    plt.subplot(2, 1, 1)
    plt.title("Original Image Histogram")
    plt.hist(gray_image)
    plt.subplot(2, 1, 2)
    plt.title("Noised Image Histogram")
    plt.hist(noise_img)
    plt.show()
    exit()


if __name__ == "__main__":
    # test()
    parser = argparse.ArgumentParser(description="HW4")
    utils.parserAdd_general(parser)
    hw1.parserAdd_hw1(parser)
    hw2.parserAdd_hw2(parser)
    hw3.parserAdd_hw3(parser)
    parserAdd_hw4(parser)
    utils.orderRun(parser)
