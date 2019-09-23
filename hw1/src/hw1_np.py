"""
Author: linnil1
Objective: Image Processing HW1
Description: This program 1)read a spectial format called 64, which represented 
a image, 2)do some operation (multiply, add, avg) on it and 3)draw histogram.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils


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
    Add the image by a constant. Make sure the value in my image is from 0 to 1.
    """
    return img + num / 256


@limitImg
def imageAvg(img1, img2):
    """
    Make average of two image pixelwisely.
    """
    return (img1 + img2) / 2


@limitImg
def image_special_func(img):
    """
    Special operation to this image
    """
    return img[:, :] - img[:, ::-1] + .5


def test():
    # read
    # real_image = read64("LINCOLN.64")
    # real_image = read64("JET.64")
    real_image = utils.read64("../LIBERTY.64")
    real_image1 = utils.read64("../LISA.64")

    # now_img = imageAvg(real_image, real_image1)
    now_img = imageAdd(real_image, 100)

    # plot it
    plt.figure()
    plt.imshow(now_img, cmap="gray")
    # histogram
    hist_bin = utils.getHist(now_img)
    print("Separate 0~1 to 32 bins")
    print(hist_bin)
    plt.figure()
    plt.bar(np.arange(32), height=hist_bin)
    # built-in histogram
    # plt.figure()
    # plt.hist(np.int8(real_image.flatten() * 31), bins=32)
    plt.show()


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
