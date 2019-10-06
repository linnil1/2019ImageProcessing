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
import cv2
import argparse
import utils
OrderAction = utils.OrderAction


def readRGB(filename):
    """
    The only method to read RGB file by cv2
    """
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(img) / 255


def getHist(img):
    """
    Calculate histogram in image
    """
    gray255_image = np.array(img * 255, dtype=np.int)
    gray255_image[gray255_image > 255] = 1
    arr_count = np.zeros(256)
    bincount = np.bincount(gray255_image.flatten())
    arr_count[:bincount.size] = bincount
    return arr_count / arr_count.sum()


def toGrayA(img):
    """
    Gray Scale: (R + G + B) / 3
    """
    if len(img.shape) == 2:
        return img
    return img.mean(2)


def toGrayB(img):
    """
    Gray Scale: 0.299R + 0.587G + 0.114B
    """
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


def resizeFromStr(img, res):
    if len(img.shape) == 2:
        return bilinear(img, [int(i) for i in res.split('x')])
    else:
        return np.stack([resizeFromStr(img[:, :, 0], res),
                         resizeFromStr(img[:, :, 1], res),
                         resizeFromStr(img[:, :, 2], res)], 2)

def histogramEqualize(img):
    """
    Perform histogram Equalization
    """
    gray255_image = np.array(img * 255, dtype=np.int)
    gray255_image[gray255_image > 255] = 1

    count_old = getHist(img)
    map_value = np.cumsum(count_old)
    return map_value[gray255_image]


def gammaCorrection(img, num):
    """
    Perform gamma correction
    """
    return img ** num


def showHist(img):
    return utils.showHist(getHist(img))


def test():
    # read
    real_image = readRGB("data/kemono_friends.jpg")
    plt.imshow(gammaCorrection(real_image, 2))
    plt.show()


def parserAdd_hw2(parser):
    parser.add_argument('--read',      type=str,   help="The image you want to read",
                        func=readRGB,   layer=(0, 1),   action=OrderAction)
    parser.add_argument('--graya',     nargs=0,        help="Convert to gray scale by A method",
                        func=toGrayA,                   action=OrderAction)
    parser.add_argument('--grayb',     nargs=0,    help="Convert to gray scale by B method",
                        func=toGrayB,                   action=OrderAction)
    parser.add_argument('--histogram', nargs=0,    help="Display Histogram",
                        func=showHist, layer=(1, None), action=OrderAction)
    parser.add_argument('--threshold', type=float, help="Set Threshold to make the grayscale image binary",
                        func=setThreshold,              action=OrderAction)
    parser.add_argument('--resize',    type=str,   help="Resize the image to XxY. Usage: --resize 1000x1000",
                        func=resizeFromStr,             action=OrderAction)
    parser.add_argument('--equalize',  nargs=0,    help="Perform histogram equalization",
                        func=histogramEqualize,         action=OrderAction)
    parser.add_argument('--gamma',     type=float, help="Perform gamma correction",
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
