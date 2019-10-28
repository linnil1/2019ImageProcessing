"""
Author: linnil1
Objective: Image Processing HW4
Description: Some operations are implemented in the file
* Homomorphic filter
* Wiener filter
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


def homomorphic(img, f_low, f_high, cutoff, c=1):
    """ Homomorphic Filter """
    def homoFilter(dest):
        return f_low + (f_high - f_low) * (1 - np.exp(-c * dest / cutoff ** 2))
    return np.exp(hw3.feqOperation(np.log(img + 1), homoFilter) - 1)


def parserAdd_hw4(parser):
    parser.add_argument("--homomorphic",  type=float,
                        metavar=("f_low", "f_high", "cutoff"), nargs=3,
                        func=homomorphic, action=OrderAction)


def test():
    # read
    real_image = hw2.readRGB("../hw3/data/Image 3-3.jpg")
    gray_image = hw2.toGrayA(real_image)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(gray_image, cmap="gray")

    fft_image = np.fft.fft2(gray_image)
    back_image = np.real(np.fft.ifft2(fft_image))

    plt.subplot(1, 3, 2)
    plt.title(f"fft -> ifft")
    plt.imshow(back_image, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title(f"Difference")
    plt.imshow(gray_image - back_image, cmap="gray")

    plt.show()
    exit()


if __name__ == "__main__":
    test()
    parser = argparse.ArgumentParser(description="HW4")
    utils.parserAdd_general(parser)
    hw1.parserAdd_hw1(parser)
    hw2.parserAdd_hw2(parser)
    hw3.parserAdd_hw3(parser)
    parserAdd_hw4(parser)
    utils.orderRun(parser)
