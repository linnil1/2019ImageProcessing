import numpy as np
import matplotlib.pyplot as plt
import utils
from utils import transform, linear, \
        transX, transY, scaleX, scaleY


def geoTransform(img, want_mask):
    """ Geometric transform """
    # Make mask same x shape with image
    x, y = np.where(want_mask)
    affine = scaleX(img.shape[0] / (x.max() - x.min() + 1)) * transX(-x.min())
    want_fullx = transform(want_mask, affine, new_shape=img.shape)

    # Resize the image by row
    img_pad = np.pad(img, [[0, 0], [0, 1]])
    want_fullx_img = np.zeros(want_fullx.shape)
    for i in range(img.shape[0]):
        print(i)
        padrow = img_pad[i]
        y = np.where(want_fullx[i])[0]
        corr_y = (y - y.min()) / (y.max() - y.min()) * (img.shape[1] - 1)
        int_y = corr_y.astype(np.int)
        want_fullx_img[i, y] = linear(corr_y,
                                      padrow[int_y], padrow[int_y + 1])

    # Inverse affine transform to original shape of mask
    want_img = transform(want_fullx_img, np.linalg.inv(affine))
    want_img = want_img[:want_mask.shape[0], :want_mask.shape[1]]
    """
    plt.subplot(221)
    plt.title("Target binary mask")
    plt.imshow(want_mask, cmap="gray")

    plt.subplot(222)
    plt.title("Affine transform to same x shape")
    plt.imshow(want_fullx, cmap="gray")

    plt.subplot(223)
    plt.title("Resize to image to mask")
    plt.imshow(want_fullx_img, cmap="gray")

    plt.subplot(224)
    plt.title("Affine transform back to same x shape")
    plt.imshow(want_img, cmap="gray")
    plt.show()
    """
    return want_img


# Set target image
img = plt.imread("data/part1/IP_dog.bmp")[:, :, 0] / 255
want_mask = np.zeros([1000, 1000])
y, x = np.meshgrid(np.arange(want_mask.shape[0]),
                   np.arange(want_mask.shape[1]))
want_mask[(x - 500) ** 2 * 3 + (y - 500) ** 2 < 450 ** 2] = 1
plt.imsave("data/mask.png", want_mask, cmap="gray")
"""
r = 20
want_mask[:, :] = 1
for i in range(0, 1100, r * 2):
    want_mask[(x - i) ** 2 + (y - 0) ** 2 / 4 < r ** 2] = 0
    want_mask[(x - 0) ** 2 / 4 + (y - i) ** 2 < r ** 2] = 0
    want_mask[(x - i) ** 2 + (y - 1000) ** 2 / 4 < r ** 2] = 0
    want_mask[(x - 1000) ** 2 / 4+ (y - i) ** 2 < r ** 2] = 0
"""

plt.subplot(121)
plt.title("Target binary mask")
plt.imshow(want_mask, cmap="gray")
plt.subplot(122)
plt.title("result")
want_img = geoTransform(img, want_mask)
plt.imshow(want_img, cmap="gray")
plt.show()
