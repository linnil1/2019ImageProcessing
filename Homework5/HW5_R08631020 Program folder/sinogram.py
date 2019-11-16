import numpy as np
import matplotlib.pyplot as plt
import argparse
import hw1_np as hw1
import hw2_np as hw2
import hw3_np as hw3
import utils
from functools import wraps


def linear(q, v1, v2):
    return v1 + (q - np.array(q, dtype=np.int)) * (v2 - v1)


def transform(img, affine):
    """
    Affine Transform
    """
    # get locations of all points in new image
    new_img = np.zeros(img.shape)
    y, x = np.meshgrid(np.arange(new_img.shape[1]),
                       np.arange(new_img.shape[0]))
    z = np.ones(new_img.shape)
    xyz = np.stack([x, y, z], 2)

    # get new locations
    affine = np.array(affine ** -1)
    pos = xyz.dot(affine.T)

    # get nonzero
    avail = (0 <= pos[:, :, 0]) & (pos[:, :, 0] < img.shape[0]) & \
            (0 <= pos[:, :, 1]) & (pos[:, :, 1] < img.shape[1])
    pos_avail = pos[avail]

    # add padding that ensure not larger than border
    data = np.pad(img, ((1, 2), (1, 2)), 'constant')
    int_x = np.array(pos_avail[:, 0], dtype=np.int32) + 1
    int_y = np.array(pos_avail[:, 1], dtype=np.int32) + 1

    # bilinear
    new_img[avail] = linear(pos_avail[:, 0],
                            linear(pos_avail[:, 1],
                                   data[int_x,     int_y],
                                   data[int_x,     int_y + 1]),
                            linear(pos_avail[:, 1],
                                   data[int_x + 1, int_y],
                                   data[int_x + 1, int_y + 1]))
    return new_img


def rotate(th):
    """ Rotated metric """
    th *= np.pi / 180
    m = np.matrix(np.zeros([3, 3]))
    m[2, 2] = 1
    m[0, 0] = m[1, 1] = np.cos(th)
    m[0, 1] = -np.sin(th)
    m[1, 0] = np.sin(th)
    return m


def setMetrix(*loc):
    """ Basic metric """
    m = np.matrix(np.zeros([3, 3]))
    m[0, 0] = m[1, 1] = m[2, 2] = 1

    def wrap(r=1):
        m[loc] = r
        return m
    return wrap


# default matrix
Base   = setMetrix(2, 2)
transX = setMetrix(0, 2)
transY = setMetrix(1, 2)


def rotateImg(img, th):
    """ Rotate the image by degree """
    s = img.shape[0] // 2
    affine = Base()     * transX( s) * transY( s) * \
             rotate(th) * transX(-s) * transY(-s)
    return transform(img, affine)


def sinoGet(img, n=181):
    """ Get sinogram """
    # pad to odd square
    s = np.max(img.shape)
    s += s % 2 == 0
    pad = s // 2 - np.array(img.shape) // 2
    img = np.pad(img, list(zip(pad, -pad + s - img.shape)), "constant")

    # get sinogram angle by angle
    sinogram = []
    for i in np.linspace(0, 180, n):
        new_img = rotateImg(img, i)
        sinogram.append(np.mean(new_img, axis=0))
    return np.array(sinogram)


def basicBackProjection(sinogram):
    """ Back projection by addition """
    n = sinogram.shape[0] - 1
    restore_img = np.zeros([sinogram.shape[1]] * 2)
    for i in range(n):
        print(i)
        rep_img = np.tile(sinogram[i], [sinogram.shape[1], 1])
        new_img = rotateImg(rep_img, i * 180 / n)
        restore_img += new_img
    return restore_img / n


def fouierBackProjection(sinogram, func):
    img_feq = np.fft.fftshift(np.fft.fft(sinogram))
    w = func(sinogram.shape[1])
    new_img = np.fft.ifft(np.fft.ifftshift(img_feq * w))
    restore_img = basicBackProjection(new_img)
    restore_img[restore_img < 0] = 0
    return restore_img


def rampFunc(n):
    """ Retrun ramp func """
    return np.abs(np.arange(n) - n // 2)


def hammingWindow(n, c):
    """
    return a hamming window.
    Usage: fouierBackProjection(sinogram,
               lambda n: rampFunc(n) * hammingWindow(n, 0.5))
    """
    return c + (c - 1) * np.cos(2 * np.pi * np.arange(n) / n)


# basic
real_image = hw2.readRGB("data/HW05-2-01.bmp")
img = hw2.toGrayA(real_image)

# easy img
# img = np.zeros([100, 100])
# img[40:60, 20:80] = 1

# get sinogram and restore back
sinogram = sinoGet(img)
np.save("data/sino_sample1.npy", sinogram)
sinogram = np.load("data/sino_sample1.npy")
# restore_img = basicBackProjection(sinogram)
restore_img = fouierBackProjection(sinogram, rampFunc)
# restore_img = fouierBackProjection(sinogram,
#                   lambda n: rampFunc(n) * hammingWindow(n, 0.5))

# plot
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.subplot(132)
plt.title("Sinogram")
plt.imshow(sinogram, cmap="gray", aspect="auto")
plt.subplot(133)
plt.title("Back Projection")
plt.imshow(restore_img, cmap="gray")
plt.show()
