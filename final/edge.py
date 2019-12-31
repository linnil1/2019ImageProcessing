from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

np.random.seed(123)
img_dir = "./data_img"
files = np.random.choice(os.listdir(img_dir), size=10, replace=False)
imgs = []
skes = []

for f in files:
    # read image
    img = Image.open(os.path.join(img_dir, f)).convert('RGB')

    # edge detection
    image_edge = img.filter(ImageFilter.FIND_EDGES).convert("L")
    image_edge = np.array(image_edge)
    image_edge[image_edge < 50] = 0
    image_edge = Image.fromarray(image_edge)

    # dilate
    image_dilate = image_edge.filter(ImageFilter.MaxFilter(3))

    # concate
    image_dilate = image_dilate.convert("RGB")
    skes.append(image_dilate)
    imgs.append(img)

# save
img = np.concatenate(imgs, axis=1)
ske = np.concatenate(skes, axis=1)
a = np.concatenate([img, ske], axis=0)
plt.imsave("pack_4.jpg", a)
plt.imshow(a)
plt.show()
