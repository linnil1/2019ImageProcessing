import os
import numpy as np
from glob import glob
from PIL import Image, ImageOps



dataset_path = './'
drawA_path = os.path.join(dataset_path, 'data_draw')
drawB_path = os.path.join(dataset_path, 'data_img')


def shuffle_data(da, db):
    a_idx = list(range(len(da)))
    np.random.shuffle(a_idx )

    b_idx = list(range(len(db)))
    np.random.shuffle(b_idx)

    shuffled_da = np.array(da)[a_idx]
    shuffled_db = np.array(db)[b_idx]

    return shuffled_da, shuffled_db


def read_images(filenames, domain=None, image_size=64):
    images = []
    for fn in filenames:
        image = Image.open(fn)
        if image is None:
            continue

        h = image_size
        w = int(h * image.size[0] / image.size[1])

        image = image.resize((w, h), resample=Image.BILINEAR)
        # image = image.resize((h, h), resample=Image.BILINEAR)
        if len(image.getbands()) != 3:
            image = ImageOps.invert(image)
            image = image.convert("RGB")
        image = np.asarray(image).astype(np.float32) / 255.
        image = image.transpose(2,0,1)
        images.append( image )

    images = np.stack( images )
    return images


def get_draw_files(test=False, n_test=200):
    dataA = glob(os.path.join(drawA_path, '*.jpg'))
    dataB = glob(os.path.join(drawB_path, '*.jpg'))
    print("dataA", len(dataA))
    print("dataB", len(dataB))

    if test == False:
        return dataA[:-n_test], dataB[:-n_test]
    if test == True:
        return dataA[-n_test:], dataB[-n_test:]
