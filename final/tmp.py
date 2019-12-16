from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import os
import json
import numpy as np

dir_name = "images/crop_1-500/"
files = os.listdir(dir_name)
for f in sorted(files):
    print(f)
    img = Image.open(dir_name + f).convert('L')
    img_inv = ImageOps.invert(img)
    img_blur = img_inv.filter(ImageFilter.GaussianBlur(51))
    img_fore = np.array(img_inv, dtype=np.float) - np.asarray(img_blur, dtype=np.float)
    img_fore[img_fore < 30] = 0
    img_fore = np.array(img_fore, dtype=np.uint8)
    img_eq = ImageOps.equalize(Image.fromarray(img_fore, mode="L"))
    img_eqblur = img_eq.filter(ImageFilter.GaussianBlur(1))
    img_back = ImageOps.invert(img_eq)
    img_back = np.array(img_back, dtype=np.uint)
    img_smoo = ImageOps.invert(img_eqblur)
    img_smoo = np.array(img_smoo, dtype=np.uint)

    plt.figure(figsize=(16, 9))
    plt.subplot(141)
    plt.title("Original")
    plt.imshow(img, cmap="gray")
    plt.subplot(142)
    plt.title("Remove backgroud")
    plt.imshow(ImageOps.invert(Image.fromarray(img_fore, mode="L")), cmap="gray")
    plt.subplot(143)
    plt.title("Histogram Equalize")
    plt.imshow(img_back, cmap="gray")
    plt.subplot(144)
    plt.title("Smooth the line")
    plt.imshow(img_smoo, cmap="gray")
    plt.show()
