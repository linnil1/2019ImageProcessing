import os
import numpy as np
from glob import glob
from PIL import Image, ImageOps
from torch.autograd import Variable
from torch import FloatTensor


dataset_path = './'
drawA_path = os.path.join(dataset_path, 'data_draw')
drawB_path = os.path.join(dataset_path, 'data_img')


def imageTensor(da, db):
    da = da.transpose(0, 3, 1, 2)
    db = db.transpose(0, 3, 1, 2)
    da = Variable( FloatTensor( da ), requires_grad=False) / 256
    db = Variable( FloatTensor( db ), requires_grad=False) / 256
    return  da, db



def shuffle_data(da, db):
    la, lb = len(da), len(db)

    a_idx = list(range(la))
    np.random.shuffle(a_idx )
    b_idx = list(range(lb))
    np.random.shuffle(b_idx)

    shuffled_da = da[a_idx]
    shuffled_db = db[b_idx]

    # randomm flip
    for i in range(0, la // 2):
        shuffled_da[i] = Image.fromarray(shuffled_da[i]).transpose(Image.FLIP_LEFT_RIGHT)
    for i in range(lb // 4, lb * 3 // 4):
        shuffled_db[i] = Image.fromarray(shuffled_db[i]).transpose(Image.FLIP_LEFT_RIGHT)

    return shuffled_da, shuffled_db


def read_images(filenames, domain=None, image_size=64, aspect=True, invert=True):
    images = []
    for fn in filenames:
        image = Image.open(fn)
        if image is None:
            continue

        h = image_size
        if aspect:
            w = int(h * image.size[0] / image.size[1])
            image = image.resize((w, h), resample=Image.BILINEAR)
        else:
            image = image.resize((h, h), resample=Image.BILINEAR)
        if len(image.getbands()) != 3:
            if invert:
                image = ImageOps.invert(image)
            image = image.convert("RGB")
        # image = np.asarray(image).astype(np.float32) / 255.
        # image = image.transpose(2,0,1)
        images.append( image )

    images = np.stack( images )
    return images


def get_draw_files(test=False, n_test=200):
    dataA = glob(os.path.join(drawA_path, '*.jpg'))
    dataB = glob(os.path.join(drawB_path, '*.jpg'))
    print("dataA", len(dataA))
    print("dataB", len(dataB))

    # shuffle for testing
    a_idx = list(range(len(dataA)))
    np.random.shuffle(a_idx)
    b_idx = list(range(len(dataB)))
    np.random.shuffle(b_idx)
    dataA = np.array(dataA)[a_idx]
    dataB = np.array(dataB)[b_idx]

    if test == False:
        return dataA[:-n_test], dataB[:-n_test]
    if test == True:
        return dataA[-n_test:], dataB[-n_test:]
