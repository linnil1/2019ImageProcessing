"""
Author: linnil1
Objective: Image Processing HW6
Description: Some operations are implemented in the file
* Wavelet transform, fusion
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
import hw1_np as hw1
import hw2_np as hw2
import hw3_np as hw3
import hw4_np as hw4
import hw5_np as hw5
import utils_cpp
import utils
from utils import OrderAction
from functools import wraps
import enum
from enum import Enum


# harr basic
harr_scale = np.array([1, 1]) / np.sqrt(2)
harr_wavel = np.array([1, -1]) / np.sqrt(2)
harr_scale_r = np.flip(harr_scale)
harr_wavel_r = np.flip(harr_wavel)


def padTo2power(img):
    """ Make the shape of image to 2**n """
    s = np.array(2 ** np.ceil(np.log2(img.shape)), dtype=np.int)
    a = s - img.shape
    return np.pad(img, [[0, a[0]], [0, a[1]]])


def upSample(data):
    """ Upsameple the 1D array and add 0 between data """
    z = np.zeros([data.shape[0], data.shape[1] * 2])
    z[:, ::2] = data
    return z


def convolve(data, f):
    """
    1D convolve function apply across 2D
    """
    return np.stack([np.convolve(d, f) for d in data])


def wavelet2D(data, depth):
    """ Wavelet transform with Harr """
    if not depth:
        return data

    # by column
    scale = convolve(data.T, harr_scale_r)[:, 1::2].T
    wavel = convolve(data.T, harr_wavel_r)[:, 1::2].T

    # by row
    scale_scale = convolve(scale, harr_scale_r)[:, 1::2]
    wavel_h     = convolve(scale, harr_wavel_r)[:, 1::2]
    wavel_v     = convolve(wavel, harr_scale_r)[:, 1::2]
    wavel_wavel = convolve(wavel, harr_wavel_r)[:, 1::2]

    # recursion
    scale_scale = wavelet2D(scale_scale, depth - 1)
    return np.vstack([np.hstack([scale_scale, wavel_h]),
                      np.hstack([wavel_v, wavel_wavel])])


def wavelet2DInv(data, depth):
    """ Inversed Wavelet transform with Harr """
    if not depth:
        return data
    h, w = np.array(data.shape) // 2

    # recursion
    scale_scale = wavelet2DInv(data[:h, :w], depth - 1)

    # by row
    scale_scale = convolve(upSample( scale_scale), harr_scale)[:, :-1]
    wave_h      = convolve(upSample(data[:h, w:]), harr_wavel)[:, :-1]
    wave_v      = convolve(upSample(data[h:, :w]), harr_scale)[:, :-1]
    wavel_wavel = convolve(upSample(data[h:, w:]), harr_wavel)[:, :-1]

    # by column
    scale = convolve(upSample((scale_scale + wave_h).T), harr_scale)[:, :-1].T
    wavel = convolve(upSample((wavel_wavel + wave_v).T), harr_wavel)[:, :-1].T
    return wavel + scale


def fusion(img1, img2, depth):
    """ Fuse two image together by wavelet """
    imgs = [img1, img2]
    ori_size = imgs[0].shape
    if any([img.shape != ori_size for img in imgs]):
        raise ValueError("Different image size to fusion")

    # wavelet
    imgs = [padTo2power(img) for img in imgs]
    print([img.shape for img in imgs])
    wave_img = [wavelet2D(img, depth) for img in imgs]

    # fusion (Compare abs value)
    subsize = np.array(imgs[0].shape) // 2 ** depth
    minmax = np.min(wave_img, axis=0), np.max(wave_img, axis=0)
    v = np.argmax(np.abs(minmax), axis=0)
    new_img = minmax[1]
    new_img[v == 0] = minmax[0][v == 0]
    new_img[:subsize[0], :subsize[1]] = \
        np.mean(wave_img, axis=0)[:subsize[0], :subsize[1]]
    return wavelet2DInv(new_img, depth)[:ori_size[0], :ori_size[1]]


def parserAdd_hw6(parser):
    parser.add_argument("--wavelet",         type=int, metavar=("depth"),
                        func=wavelet2D,      action=OrderAction)
    parser.add_argument("--waveletinv",      type=int, metavar=("depth"),
                        func=wavelet2DInv,   action=OrderAction)
    parser.add_argument("--fusion",          type=int, metavar=("depth"),
                        func=fusion,         action=OrderAction, layer=(2, 1))


def test():
    # read
    # real_image = hw2.readRGB("../hw3/data/Image 3-3.jpg")
    # img = hw2.readRGB("../hw2/data/kemono_friends.jpg")
    # img = hw2.readRGB("data/HW05-3-03.bmp")
    # imgs = ["data/part2/set1/clock1.JPG", "data/part2/set1/clock2.JPG"]
    # imgs = ["data/part2/set2/multifocus1.JPG", "data/part2/set2/multifocus2.JPG", "data/part2/set2/multifocus3.JPG"]
    imgs = ["data/part2/set3/MRI1.jpg", "data/part2/set3/MRI2.jpg"]
    imgs = [plt.imread(name)[:,:] / 255 for name in imgs]

    for i in range(3):
        if i < len(imgs):
            plt.subplot(231 + 2 * i)
            plt.imshow(imgs[i], cmap="gray")

        plt.subplot(234 + i)
        dep = i + 1
        restore_img = fusion(*imgs, dep)
        plt.title(f"Wavelet fusion with depth={dep}")
        plt.imshow(restore_img, cmap="gray")

    plt.show()
    exit()


if __name__ == "__main__":
    # testTransform()
    test()
    parser = argparse.ArgumentParser(description="HW6")
    utils.parserAdd_general(parser)
    hw1.parserAdd_hw1(parser)
    hw2.parserAdd_hw2(parser)
    hw3.parserAdd_hw3(parser)
    hw4.parserAdd_hw4(parser)
    hw5.parserAdd_hw5(parser)
    utils.orderRun(parser)
