"""
Author: linnil1
Objective: Image Processing HW5
Description: Some operations are implemented in the file
* Color Space transform: RGB, HSI, CMY, CMYK
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
import enum
from enum import Enum


class Color(Enum):
    RGB = "RGB"
    CMY = "CMY"
    CMYK = "CMYK"
    HSI = "HSI"
    XYZ = "XYZ"
    LAB = "LAB"


# Textbook: Gonzalez, R. C., & Woods, R. E. (2017). Digital image processing, 4th edn. ISBN: 9780133356724.
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
        return np.concatenate([(cmy - k) /  (1 - k), k], axis=2)

    elif space == Color.HSI:
        # Textbook
        i = np.mean(img, axis=2)
        s = 1 - np.min(img, axis=2) / (i + 1e-6)
        s[s < 0] = 0
        R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
        th = np.arccos((2 * R - G - B) / 2 / (
             np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6))
        h = th
        h[B > G] = 2 * np.pi - h[B > G]
        assert((i >= 0).all())
        assert(((1 >= s) & (s >= 0)).all())
        assert(((2 * np.pi >= h) & (h >= 0)).all())
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
        XYZ = fromRGB(Color.XYZ) / XYZ_ref
        XYZ_f = np.zeros(XYZ.shape)
        i = XYZ > 0.008856
        XYZ_f[i] = XYZ[i] ** (1 / 3)
        XYZ_f[~i] = 7.787 * XYZ[~i] + 16 / 116

        L = 116 * XYZ_f[:, :, 0] - 16
        a = 500 * (XYZ_f[:, :, 0] - XYZ_f[:, :, 1])
        b = 200 * (XYZ_f[:, :, 1] - XYZ_f[:, :, 2])
        return np.stack([L, a, b], axis=2)


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
        h, s, i = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        R = np.zeros(img.shape[:2])
        G, B = R.copy(), R.copy()

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

    else:
        raise ValueError("No such format {space}")


def colorTransform(img, space1, space2):
    img_rgb = img
    if space1 != Color.RGB:
        img_rgb = toRGB(img, space1)
    return fromRGB(img_rgb, space2)


def test():
    # read
    # real_image = hw2.readRGB("../hw3/data/Image 3-3.jpg")
    img = hw2.readRGB("../hw2/data/kemono_friends.jpg")
    new_img = colorTransform(img, Color.RGB, Color.XYZ)
    # new_img[:,:,2] -= .2
    print(img[:5, :5])
    print(new_img[:5, :5])
    print(new_img.shape)
    new_img = colorTransform(new_img, Color.XYZ, Color.RGB)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.title("Subtract I with HSI")
    plt.imshow(new_img)
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
