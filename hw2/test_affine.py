import matplotlib.pyplot as plt
import numpy as np


def linear(q, v1, v2):
    return v1 + (q - np.array(q, dtype=np.int)) * (v2 - v1)


def transform(img, affine, add_size=(0, 0)):
    # get locations of all points in new image
    new_img = np.zeros(np.array(img.shape) + add_size)
    y, x = np.meshgrid(np.arange(new_img.shape[1]),
                       np.arange(new_img.shape[0]))
    z = np.ones(new_img.shape)
    xyz = np.stack([x, y, z], 2)

    # get new locations
    print("ori:\n", affine)
    affine = np.array(affine ** -1).T
    pos = xyz.dot(affine)
    print("inv:\n", affine)

    # get nonzero
    avail = (0 <= pos[:, :, 0]) & (pos[:, :, 0] < img.shape[0]) & \
            (0 <= pos[:, :, 1]) & (pos[:, :, 1] < img.shape[1])
    pos_avail = pos[avail]

    # add padding and it's left top corner
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
    th *= np.pi / 180
    m = np.matrix(np.zeros([3, 3]))
    m[2, 2] = 1
    m[0, 0] = m[1, 1] = np.cos(th)
    m[0, 1] = -np.sin(th)
    m[1, 0] = np.sin(th)
    return m


def setMetrix(*loc):
    m = np.matrix(np.zeros([3, 3]))
    m[0, 0] = m[1, 1] = m[2, 2] = 1

    def wrap(r=1):
        m[loc] = r
        return m
    return wrap


# basic
img = plt.imread("data/kemono_friends.jpg").mean(2)
Base   = setMetrix(2, 2)
transX = setMetrix(0, 2)
transY = setMetrix(1, 2)
shearX = setMetrix(0, 1)
shearY = setMetrix(1, 0)
scaleX = setMetrix(0, 0)
scaleY = setMetrix(1, 1)


# Order matters
affine4 = Base() * scaleX(10) * transX(50)
affine5 = Base() * transX(50) * scaleX(10)
print("Order Matter:", affine4 - affine5)

# plot three operations
affine = Base()
new_img = transform(img, affine, (100, 100))
plt.subplot(221)
plt.title("Original")
plt.imshow(new_img, cmap="gray")

affine = Base() * scaleX(0.8) * scaleY(1.2) * transX(80) * transY(12)
new_img = transform(img, affine, (100, 100))
plt.subplot(223)
plt.title("Scale + translate")
plt.imshow(new_img, cmap="gray")

affine = Base() * scaleX(.8) * scaleY(.8) * \
         transX(250) * transY(50) * rotate(45)
new_img = transform(img, affine, (100, 100))
plt.subplot(224)
plt.title("Scale + translate + rotate")
plt.imshow(new_img, cmap="gray")

affine = Base() * scaleX(.8) * scaleY(.8) * \
         transX(50) * transY(50) * shearX(.2) * rotate(-20)
new_img = transform(img, affine, (100, 100))
plt.subplot(222)
plt.title("Scale + translate + rotate + shearX")
plt.imshow(new_img, cmap="gray")

plt.show()
