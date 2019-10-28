"""
Author: linnil1
Objective: Image Processing HW2
Description: Some operations are implemented in the file
* read color image
* convert to gray scale
* resize
* get histogram
* show histogram
* histogram equalization
* gamma correction
* Binarize by threshold
"""

import numpy as np
import matplotlib.pyplot as plt
# import cv2
import argparse
import utils
import re
OrderAction = utils.OrderAction


def readRGB(filename):
    """ Read: Color image """
    # img = cv2.imread(filename)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = plt.imread(filename)
    return np.array(img) / 255


def getHist(img):
    """ Histogram """
    gray255_image = np.array(img * 255, dtype=np.int)
    gray255_image[gray255_image > 255] = 255
    arr_count = np.zeros(256)
    bincount = np.bincount(gray255_image.flatten())
    arr_count[:bincount.size] = bincount
    return arr_count / arr_count.sum()


def toGrayA(img):
    """ Gray Scale: (R + G + B) / 3 """
    if len(img.shape) == 2:
        return img
    return img.mean(2)


def toGrayB(img):
    """ Gray Scale: 299R + 587G + 114B """
    if len(img.shape) == 2:
        return img
    return img.dot([0.299, 0.587, 0.114])


def setThreshold(img, threshold):
    """
    Set threshold to binarize image.
    Input image should be gray scale.
    """
    return img > threshold / 255


def linear(q, v1, v2):
    if q.shape == v1.shape:
        return v1 + q[:, :] * (v2 - v1)
    else:
        return v1 + q[:, :, None] * (v2 - v1)


def bilinear(img, new_shape):
    """ Resize by bilinear interpolation """
    if new_shape is str:
        new_shape = utils.parseSize(res)
    # prepare data
    print(f"{img.shape} -> {new_shape}")
    if len(img.shape) == 2:
        data = np.pad(img, ((0, 1), (0, 1)), "constant")
    else:
        data = np.pad(img, ((0, 1), (0, 1), (0, 0)), "constant")

    # prepare x,y
    ksizex = img.shape[0] / new_shape[0]
    ksizey = img.shape[1] / new_shape[1]
    y, x = np.meshgrid(np.arange(new_shape[1]) * ksizey,
                       np.arange(new_shape[0]) * ksizex)
    int_x = np.array(x, dtype=np.int32)
    int_y = np.array(y, dtype=np.int32)

    # bilinear
    return linear(x - int_x,
                  linear(y - int_y,
                         data[int_x,     int_y],
                         data[int_x,     int_y + 1]),
                  linear(y - int_y,
                         data[int_x + 1, int_y],
                         data[int_x + 1, int_y + 1]))


def histogramEqualize(img):
    """ Transformation: Histogram Equalization """
    gray255_image = np.array(img * 255, dtype=np.int)
    gray255_image[gray255_image > 255] = 1

    count_old = getHist(img)
    map_value = np.cumsum(count_old)
    return map_value[gray255_image]


def gammaCorrection(img, num):
    """ Transformation: Gamma correction """
    return img ** num


def showHist(img):
    """ Display histogram """
    return utils.showHist(getHist(img))


def test():
    # read
    real_image = readRGB("data/kemono_friends.jpg")
    plt.imshow(gammaCorrection(real_image, 2))
    plt.show()


def parserAdd_hw2(parser):
    parser.add_argument("--read",      type=str,        metavar=("ImageFilePath"),
                        func=readRGB,  layer=(0, 1),    action=OrderAction)
    parser.add_argument("--graya",     nargs=0,
                        func=toGrayA,                   action=OrderAction)
    parser.add_argument("--grayb",     nargs=0,
                        func=toGrayB,                   action=OrderAction)
    parser.add_argument("--histogram", nargs=0,
                        func=showHist, layer=(1, None), action=OrderAction)
    parser.add_argument("--threshold", type=float,
                        func=setThreshold,              action=OrderAction)
    parser.add_argument("--resize",    type=str,        metavar=("aaaxbbb"),
                        func=bilinear,                  action=OrderAction)
    parser.add_argument("--equalize",  nargs=0,
                        func=histogramEqualize,         action=OrderAction)
    parser.add_argument("--gamma",     type=float,
                        func=gammaCorrection,           action=OrderAction)


if __name__ == "__main__":
    # test()
    # exit()
    import hw1_np as hw1
    parser = argparse.ArgumentParser(description="HW2")
    utils.parserAdd_general(parser)
    hw1.parserAdd_hw1(parser)
    parserAdd_hw2(parser)
    utils.orderRun(parser)
