from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import os
import json
import numpy as np

dir_name = "images/crop_1-500/"
dir_save = "images/clean_1-500/"

files = os.listdir(dir_name)
for f in sorted(files):
    print(f)
    # gray scale and invert
    img = Image.open(dir_name + f).convert('L')
    img_inv = ImageOps.invert(img)
    # remove backgroud
    img_blur = img_inv.filter(ImageFilter.GaussianBlur(51))
    img_fore = np.array(img_inv, dtype=np.float) - np.asarray(img_blur, dtype=np.float)
    img_fore[img_fore < 30] = 0
    img_fore = np.array(img_fore, dtype=np.uint8)
    # Line enhencing and smoothing
    img_eq = ImageOps.equalize(Image.fromarray(img_fore, mode="L"))
    img_eqblur = img_eq.filter(ImageFilter.GaussianBlur(1))
    # invert back
    img_back = ImageOps.invert(img_eqblur)

    # resize
    want_h = 1500
    want_w = int(want_h * img_back.size[0] / img_back.size[1])
    img_res = img_back.resize((want_w, want_h), resample=Image.BILINEAR)
    print(img_res.size)

    # padding
    img_pad = Image.new("L", (1200, 1600), 255)
    offset = ((img_pad.size[0] - img_res.size[0]) // 2, 50)
    img_pad.paste(img_res, offset)

    img_pad.save(dir_save + f)
    # plt.imshow(img_pad, cmap="gray")
    # plt.show()
