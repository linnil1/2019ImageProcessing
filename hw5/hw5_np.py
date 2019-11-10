"""
Author: linnil1
Objective: Image Processing HW5
Description: Some operations are implemented in the file
* Color Space transform: RGB, CMY, CMYK, HSI, XYZ, LAB, YUV
* Pseudo Color
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
import hw1_np as hw1
import hw2_np as hw2
import hw3_np as hw3
import hw4_np as hw4
import utils_cpp
import utils
from utils import OrderAction
from functools import wraps
import enum
from enum import Enum


class Color(Enum):
    RGB = "RGB"
    CMY = "CMY"
    CMYK = "CMYK"
    HSI = "HSI"
    XYZ = "XYZ"
    LAB = "LAB"
    YUV = "YUV"


# Textbook: Gonzalez, R. C., & Woods, R. E. (2017).
#           Digital image processing, 4th edn. ISBN: 9780133356724.
def fromRGB(img, space):
    """ From RGB to other space """
    if space == Color.RGB:
        # Textbook
        return img

    elif space == Color.CMY:
        # Textbook
        return 1 - img

    elif space == Color.CMYK:
        # Textbook
        cmy = fromRGB(img, Color.CMY)
        k = np.min(cmy, axis=2, keepdims=True)
        return np.concatenate([(cmy - k) / (1 - k), k], axis=2)

    elif space == Color.HSI:
        # Textbook
        i = np.mean(img, axis=2)
        s = 1 - np.min(img, axis=2) / (i + 1e-6)
        s[s < 0] = 0
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        th = np.arccos((2 * R - G - B) / 2 / (
             np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6))
        h = th
        h[B > G] = 2 * np.pi - h[B > G]
        return np.stack([h, s, i], axis=2)

    elif space == Color.XYZ:
        # https://www.cs.rit.edu/~ncs/color/t_convert.html
        m = np.array(np.matrix("""
                0.412453  0.357580  0.180423;
                0.212671  0.715160  0.072169;
                0.019334  0.119193  0.950227"""))
        return img.dot(m.T)

    elif space == Color.LAB:
        # textbook, wiki:CIELAB_color_space
        XYZ_ref = np.array([95.0489, 100, 108.8840])
        XYZ_f = fromRGB(img, Color.XYZ) / XYZ_ref
        d = 6 / 29
        i = XYZ_f > d ** 3
        XYZ_f[i] = XYZ_f[i] ** (1 / 3)
        XYZ_f[~i] = XYZ_f[~i] / (3 * d ** 2) + 4 / 29
        L = 116 * XYZ_f[:, :, 1] - 16
        a = 500 * (XYZ_f[:, :, 0] - XYZ_f[:, :, 1])
        b = 200 * (XYZ_f[:, :, 1] - XYZ_f[:, :, 2])
        return np.stack([L, a, b], axis=2)

    elif space == Color.YUV:
        # https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
        m = np.array(np.matrix("""
                0.299 0.587 0.114;
                -0.14713 -0.28886 0.436;
                0.615 -0.51499 -0.10001"""))
        return img.dot(m.T)

    else:
        raise ValueError("No such format {space}")


def toRGB(img, space):
    """
    From other space to RGB.
    Not grantee RGB is limit from 0 to 1
    """
    if space == Color.RGB:
        # textbook
        return img

    elif space == Color.CMY:
        # textbook
        return 1 - img

    elif space == Color.CMYK:
        # textbook
        cmy = img[:, :, :3] * (1 - img[:, :, 3])[:, :, None] \
            + img[:, :, 3, None]
        return toRGB(cmy, Color.CMY)

    elif space == Color.HSI:
        # textbook
        # to hsi
        h, s, i = img[:, :, 0].copy(), img[:, :, 1], img[:, :, 2]
        R = np.zeros(img.shape[:2])
        G, B = R.copy(), R.copy()
        h = h - np.floor(h / 2 / np.pi) * 2 * np.pi

        # thress sectors
        RG = h < np.pi * 2 / 3
        B[RG] = (i * (1 - s))[RG]
        R[RG] = (i * (1 + s * np.cos(h) / np.cos(np.pi / 3 - h)))[RG]
        G[RG] = (3 * i - R - B)[RG]

        h -= np.pi * 2 / 3
        GB = (0 <= h) & (h < np.pi * 2 / 3)
        R[GB] = (i * (1 - s))[GB]
        G[GB] = (i * (1 + s * np.cos(h) / np.cos(np.pi / 3 - h)))[GB]
        B[GB] = (3 * i - R - G)[GB]

        h -= np.pi * 2 / 3
        BR = (0 <= h) & (h < np.pi * 2 / 3)
        G[BR] = (i * (1 - s))[BR]
        B[BR] = (i * (1 + s * np.cos(h) / np.cos(np.pi / 3 - h)))[BR]
        R[BR] = (3 * i - G - B)[BR]

        # limit to 0 to 1
        return np.stack([R, G, B], axis=2)

    elif space == Color.XYZ:
        # https://www.cs.rit.edu/~ncs/color/t_convert.html
        m = np.array(np.matrix("""
                 3.240479 -1.537150 -0.498535;
                -0.969256  1.875992  0.041556;
                 0.055648 -0.204043  1.057311"""))
        return img.dot(m.T)

    elif space == Color.LAB:
        # textbook, wiki:CIELAB_color_space
        XYZ_ref = np.array([95.0489, 100, 108.8840])
        XYZ_inv = np.zeros(img.shape)
        XYZ_inv[:, :, 0] += img[:, :, 1] / 500
        XYZ_inv[:, :, :] += ((img[:, :, 0] + 16) / 116)[:, :, None]
        XYZ_inv[:, :, 2] -= img[:, :, 2] / 200
        d = 6 / 29
        i = XYZ_inv > d
        XYZ_inv[i] = XYZ_inv[i] ** 3
        XYZ_inv[~i] = 3 * d ** 2 * (XYZ_inv[~i] - 4 / 29)
        return toRGB(XYZ_inv * XYZ_ref, Color.XYZ)

    elif space == Color.YUV:
        # https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
        m = np.array(np.matrix("""
                1 0 1.13983;
                1 -0.39465 -0.58060;
                1 2.03211 0"""))
        return img.dot(m.T)

    else:
        raise ValueError("No such format {space}")


def colorTransform(img, space1, space2):
    """ Transform color space from space1 to space2 """
    img_rgb = img
    if space1 != Color.RGB:
        img_rgb = toRGB(img, space1)
    return fromRGB(img_rgb, space2)


def testTransform():
    """
    This plot all layers after color transform by each methods
    """
    img = hw2.readRGB("../hw2/data/kemono_friends.jpg")
    for c in list(Color):
        if c == Color.RGB:
            continue
        print(c.value)
        trans_img = fromRGB(img, c)
        n = 2 + trans_img.shape[2]

        plt.figure(figsize=(18, 6))
        plt.subplot(1, n, 1)
        plt.title("Original Image")
        plt.imshow(img)

        for i in range(trans_img.shape[2]):
            plt.subplot(1, n, 2 + i)
            plt.title(c.value[i])
            plt.imshow(trans_img[:, :, i], cmap="gray")

        plt.subplot(1, n, n)
        plt.title("Restore Image")
        back_img = hw1.limitImg01(toRGB(trans_img, c))
        plt.imshow(back_img)

        # plt.show()
        plt.savefig("data/img_" + c.value + ".png")
        plt.clf()
    exit()


def colorMap(color_start, color_end, n=256):
    """ Generate color map by starting color and ending color """
    cmap = np.zeros([n, 3])
    for i in range(3):
        cmap[:, i] = np.linspace(color_start[i], color_end[i], n)
    return cmap


def readHex(s):
    """
    Transfer hex (010203) to RGB(1, 2, 3)
    """
    v = int(s, 16)
    return np.array([v // 256 // 256, v // 256 % 256, v % 256]) / 255


def colorMapHSI(color_start, color_end, n=256):
    """
    HSI Colormap creation by provied by two colors
    """
    s, e = fromRGB(np.array([[color_start, color_end]]),
                   Color.HSI)[0]
    cmap = colorMap(s, e, n)
    return hw1.limitImg01(toRGB(cmap[None], Color.HSI)[0])


def cmPlot(img, cmap):
    """ Plot the image and it's color bar by pyplot """
    plt.figure()
    im = plt.imshow(img, cmap=ListedColormap(cmap))
    plt.colorbar(im)


def showPseudo(img, color_start, color_end, n=256):
    """ Plot pseduo color image by pyplot with hex color input"""
    cmap = colorMapHSI(readHex(color_start), readHex(color_end), n)
    plt.figure()
    im = plt.imshow(img, cmap=ListedColormap(cmap))
    plt.colorbar(im)


def createHSI(i=0.5, sizeh=256, sizes=256):
    """ Create color map for choosing """
    # H: 0 to 2\pi
    # S: 0 to 1
    # I: 0 to 1
    hsi = np.zeros([sizes, sizeh, 3])
    hsi[:, :, 0] = np.linspace(0, np.pi * 2, sizeh)
    hsi[:, :, 1] = np.linspace(1, 0, sizes)[:, None]
    hsi[:, :, 2] = i
    return toRGB(hsi, Color.HSI)


def kMean(img, k, max_epoch=1000, tor=1e-3, interger=False):
    """
    K-mean.
    The input should be 3 channels.
    The output is the image
    """
    if img.shape[-1] != 3:
        raise ValueError("Data should be three channel")
    data = img.reshape(-1, 3)
    ans = np.zeros([k, 3])
    ans = data[np.random.choice(np.arange(data.shape[0]), k, replace=False), :]

    # iterations
    for i in range(max_epoch):
        dist = np.sum((data - ans[:, None, :]) ** 2, axis=2)
        closest = np.argmin(dist, axis=0)
        now_ans = np.array([np.mean(data[closest == i], axis=0)
                            for i in range(k)])
        # stop
        if np.sum(np.abs(ans - now_ans)) < tor:
            label = closest.reshape(img.shape[:2])
            if interger:
                return label
            else:
                return label / (k - 1)
        ans = now_ans
    raise ValueError("Unable to converge")


def parserAdd_hw5(parser):
    parser.add_argument("--colortransform",  type=Color, metavar=("form", "to"), nargs=2,
                        func=colorTransform, action=OrderAction)
    parser.add_argument("--showpseudo",      type=str,   metavar=("000000", "ffffff"), nargs=2,
                        func=showPseudo,     action=OrderAction, layer=(1, None))
    parser.add_argument("--kmeans",          type=int,   metavar=("k"),
                        func=kMean,          action=OrderAction)


def test():
    # read
    # real_image = hw2.readRGB("../hw3/data/Image 3-3.jpg")
    # img = hw2.readRGB("../hw2/data/kemono_friends.jpg")
    img = hw2.readRGB("data/HW05-3-03.bmp")
    print(img.shape)
    img = hw2.bilinear(img, (68, 102))
    plt.imshow(kMean(img, 6))
    plt.show()

    color_start = [np.pi * 1.8, 1, .5]
    color_end = [np.pi * -.05, 1, 0.5]
    n = 256
    exit()


if __name__ == "__main__":
    # testTransform()
    # test()
    parser = argparse.ArgumentParser(description="HW5")
    utils.parserAdd_general(parser)
    hw1.parserAdd_hw1(parser)
    hw2.parserAdd_hw2(parser)
    hw3.parserAdd_hw3(parser)
    hw4.parserAdd_hw4(parser)
    parserAdd_hw5(parser)
    utils.orderRun(parser)
