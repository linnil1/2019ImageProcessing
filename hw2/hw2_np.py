"""
Author: linnil1
Objective: Image Processing HW2
Description:
"""

import numpy as np
import matplotlib.pyplot as plt
import utils
import cv2


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


def toGrayA(img):
    """
    Gray Scale: (R + G + B) / 3
    """
    return img.mean(2)


def toGrayB(img):
    """
    Gray Scale: 0.299R + 0.587G + 0.114B
    """
    return img.dot([0.299, 0.587, 0.114])


def setThreshold(img, threshold):
    """
    Set threshold to binarize image.
    Input image should be gray scale.
    """
    return img > threshold


def linear(q, v1, v2):
    return v1 + (q - np.int(q)) * (v2 - v1)


def bilinear(img, new_shape):
    """
    Change the shape of the image by bilinear interpoltion.
    Input image should be gray scale.
    """
    print(f"{img.shape} -> {new_shape}")
    arrz = []
    ksizex, ksizey = (np.array(img.shape) - 1) \
                   / (np.array(new_shape) - 1)
    pad_z = np.pad(img, ((0, 1), (0, 1)), 'edge')
    for i in range(new_shape[0]):
        arr = []
        now_x = i * ksizex
        int_x = np.int(now_x)
        for j in range(new_shape[1]):
            now_y = j * ksizey
            int_y = np.int(now_y)
            # for each pixel interpolate neighbor's value
            arr.append(linear(
                now_x,
                linear(now_y, pad_z[int_x,     int_y],
                              pad_z[int_x,     int_y + 1]),
                linear(now_y, pad_z[int_x + 1, int_y],
                              pad_z[int_x + 1, int_y + 1])))
        arrz.append(arr)
    return np.array(arrz)


def test():
    # read
    real_image = utils.readRGB("data/kemono_friends.jpg")

    # now_img = setThreshold(toGrayA(real_image), 168)
    now_img = bilinear(toGrayA(real_image), (1000, 1000))
    plt.imshow(now_img, cmap="gray")
    # now_img = imageAvg(real_image, real_image1)

    """
    now_img = bilinear(toGrayA(real_image), (100, 100))
    plt.subplot(121)
    plt.title("Original Image")
    plt.imshow(real_image)
    plt.subplot(122)
    plt.title("Resize to 100x100")
    plt.imshow(now_img, cmap="gray")

    gray_imga = toGrayA(real_image)
    gray_imgb = toGrayB(real_image)
    thres_imga = setThreshold(gray_imga, 168)
    thres_imgb = setThreshold(gray_imgb, 168)

    # plot it
    plt.figure()
    plt.subplot(131)
    plt.title("Original Image")
    plt.imshow(real_image)
    plt.subplot(132)
    plt.title("gray A")
    plt.imshow(gray_imga, cmap="gray")
    plt.subplot(133)
    plt.title("gray B")
    plt.imshow(gray_imgb, cmap="gray")

    plt.figure()
    plt.subplot(131)
    plt.title("Original Image")
    plt.imshow(real_image)
    plt.subplot(132)
    plt.title("A with threshold")
    plt.imshow(thres_imga, cmap="gray")
    plt.subplot(133)
    plt.title("B with threshold")
    plt.imshow(thres_imgb, cmap="gray")

    plt.figure()
    plt.subplot(121)
    plt.title("Original Image")
    plt.imshow(real_image)
    plt.subplot(122)
    plt.title("A - B")
    plt.imshow(gray_imga - gray_imgb, cmap="gray")
    """
    plt.show()


if __name__ == "__main__":
    # test()
    # exit()

    parser = utils.setParser()
    args = parser.parse_args()
    print(args)

    plt.figure()
    img = utils.readRGB(args.path)
    plt.title("Original Image")
    plt.imshow(img)

    gray_img = None
    if args.graya:
        gray_img = toGrayA(img)
    elif args.grayb:
        gray_img = toGrayB(img)

    if gray_img is not None:
        plt.figure()
        plt.title("Grayscale Image")
        plt.imshow(gray_img, cmap="gray")

        if args.threshold is not None:
            plt.figure()
            plt.title("Binarized Image")
            plt.imshow(setThreshold(gray_img, args.threshold), cmap="gray")

        if args.hist:
            plt.figure()
            hist_bin = utils.getHist(gray_img / 255)
            plt.title("Histogram")
            plt.bar(np.arange(32), height=hist_bin)

        if args.resize is not None:
            new_shape = np.array(args.resize.split('x'), dtype=np.int)
            plt.figure()
            plt.title("Resized Image")
            plt.imshow(bilinear(gray_img, new_shape), cmap="gray")

    plt.show()
