import numpy as np
import argparse
import matplotlib.pyplot as plt


def map64(ch):
    """
    Map 32 character to number
    """
    if ord('0') <= ord(ch) <= ord('9'):
        return ord(ch) - ord('0')
    elif ord('A') <= ord(ch) <= ord('Z'):
        return ord(ch) - ord('A') + 10
    else:
        return None
        raise ValueError(f"Not correct value of {ch}")


def read64(filename):
    """
    Read file that contained 64 formatted image
    """
    read_lines = open(filename).readlines()
    real_image = [[map64(ch) for ch in line if map64(ch) is not None] for line in read_lines]
    real_image = list(filter(None, real_image))
    real_image = np.array(real_image) / 31
    return real_image


def getHist(img):
    """
    Calculate histogram in image
    """
    arr = np.array([])
    arr.resize(32)
    map32 = np.uint8(img * 31).flatten()
    for i in map32:
        arr[i] += 1
    return arr / arr.sum()


def setParser():
    """
    Set argument parser
    """
    parser = argparse.ArgumentParser(description="HW1")
    parser.add_argument('path',       type=str,            help="The image you want to read") 
    parser.add_argument('--add',      type=float,          help="Add a number to the image")
    parser.add_argument('--addimage', type=str,            help="Add a image to the image")
    parser.add_argument('--multiply', type=float,          help="Multiply a number to the image")
    parser.add_argument('--special',  action="store_true", help="Operator of hw1-2-4")
    return parser


def plot(img, new_img):
    """
    plot the image
    """
    n = 3 if new_img is not None else 2
    plt.figure()
    
    # ori image
    plt.subplot(1, n, 1)
    plt.imshow(img, cmap="gray")

    # image after operations
    if new_img is not None:
        plt.subplot(1, n, 2)
        plt.imshow(new_img, cmap="gray")
        img = new_img

    # histogram
    hist_bin = getHist(img)
    print("Separate 0~1 to 32 bins")
    print(hist_bin)
    plt.subplot(1, n, n)
    plt.bar(np.arange(32), height=hist_bin)

    plt.show()
