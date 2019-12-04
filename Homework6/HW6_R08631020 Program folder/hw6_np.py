"""
Author: linnil1
Objective: Image Processing HW6
Description: Some operations are implemented in the file
* Wavelet transform, fusion
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import hw1_np as hw1
import hw2_np as hw2
import hw3_np as hw3
import hw4_np as hw4
import hw5_np as hw5
import utils_cpp
import utils
from utils import transform, linear, \
        transX, transY, scaleX, scaleY, \
        OrderAction
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


def houghTransform(img):
    """ Hough transform: The input should be edge image """
    max_dis = np.int(np.ceil(np.sqrt(np.sum(np.array(img.shape) ** 2)))) + 1

    # calculate distance by theta
    th = np.linspace(0, np.pi, 181)
    x, y = np.where(img)
    r = (np.outer(x, np.cos(th)) + np.outer(y, np.sin(th))).astype(np.int)

    # write on hough map
    hough = np.zeros([2 * max_dis, th.size])
    for i in range(len(x)):
        hough[r[i] + max_dis, np.arange(th.size)] += 1

    return hough / hough.max()


def geoTransform(img, want_mask):
    """ Geometric transform """
    # Make mask same x shape with image
    want_mask = np.int_(want_mask)
    x, y = np.where(want_mask)
    affine = scaleX(img.shape[0] / (x.max() - x.min() + 1)) * transX(-x.min()) * \
             scaleY(img.shape[1] / (y.max() - y.min() + 1)) * transY(-y.min())
    want_fullx = transform(want_mask, affine, new_shape=img.shape[:2])

    # Resize the image by row
    if len(img.shape) == 2:
        img_pad = np.pad(img, ((0, 0), (0, 1)), 'constant')
    else:
        img_pad = np.pad(img, ((0, 0), (0, 1), (0, 0)), 'constant')
    want_fullx_img = np.zeros(img.shape)
    for i in range(img.shape[0]):
        padrow = img_pad[i]
        y = np.where(want_fullx[i])[0]
        corr_y = (y - y.min()) / (y.max() - y.min()) * (img.shape[1] - 1)
        int_y = corr_y.astype(np.int)
        want_fullx_img[i, y] = linear(corr_y,
                                      padrow[int_y], padrow[int_y + 1])

    # Inverse affine transform to original shape of mask
    print(img.shape)
    print(want_fullx_img.shape)
    want_img = transform(want_fullx_img, affine ** -1, [*want_mask.shape, *img.shape[2:]])
    return want_img


def parserAdd_hw6(parser):
    parser.add_argument("--wavelet",         type=int, metavar=("depth"),
                        func=wavelet2D,      action=OrderAction)
    parser.add_argument("--waveletinv",      type=int, metavar=("depth"),
                        func=wavelet2DInv,   action=OrderAction)
    parser.add_argument("--fusion",          type=int, metavar=("depth"),
                        func=fusion,         action=OrderAction, layer=(2, 1))
    parser.add_argument("--hough",           nargs=0,
                        func=houghTransform, action=OrderAction)
    parser.add_argument("--geo",             nargs=0,
                        func=geoTransform,   action=OrderAction, layer=(2, 1))


def test():
    # read
    # img = hw2.readRGB("../hw2/data/kemono_friends.jpg")
    # imgs = ["data/part2/set1/clock1.JPG", "data/part2/set1/clock2.JPG"]
    # imgs = ["data/part2/set2/multifocus1.JPG", "data/part2/set2/multifocus2.JPG", "data/part2/set2/multifocus3.JPG"]
    # imgs = ["data/part2/set3/MRI1.jpg", "data/part2/set3/MRI2.jpg"]
    # imgs = [plt.imread(name)[:, :] / 255 for name in imgs]
    img = hw2.readRGB("data/part1/IP_dog.bmp")
    # img = img[:, :, 0]
    want_mask = np.zeros([2000, 2000])
    y, x = np.meshgrid(np.arange(want_mask.shape[0]),
                       np.arange(want_mask.shape[1]))
    want_mask[(x - 1000) ** 2 * 3 + (y - 1000) ** 2 < 550 ** 2] = 1
    want_img = geoTransform(img, want_mask)
    plt.imshow(want_img)

    plt.show()
    exit()


if __name__ == "__main__":
    # test()
    parser = argparse.ArgumentParser(description="HW6")
    utils.parserAdd_general(parser)
    hw1.parserAdd_hw1(parser)
    hw2.parserAdd_hw2(parser)
    hw3.parserAdd_hw3(parser)
    hw4.parserAdd_hw4(parser)
    hw5.parserAdd_hw5(parser)
    parserAdd_hw6(parser)
    utils.orderRun(parser)
