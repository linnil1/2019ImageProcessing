import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2


def readRGB(filename):
    """
    The only method to read RGB file by cv2
    """
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


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
    parser.add_argument('path',        type=str,            help="The image you want to read")
    parser.add_argument('--graya',     action="store_true", help="Convert to gray scale by A method")
    parser.add_argument('--grayb',     action="store_true", help="Convert to gray scale by B method")
    parser.add_argument('--hist',      action="store_true", help="Display Histogram")
    parser.add_argument('--threshold', type=float,          help="Set Threshold to make the grayscale image binary")
    parser.add_argument('--resize',    type=str,            help="Resize the image to XxY. Usage: --resize 1000x1000")
    return parser
