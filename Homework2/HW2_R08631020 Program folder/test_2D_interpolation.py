import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import timeit


def gaussian(x, mu=0, sig=1):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


nx = 20
ny = 20
knx = 80
kny = 80
ksizex = (nx - 1) / (knx - 1)
ksizey = (ny - 1) / (kny - 1)
x = np.arange(nx)
y = np.arange(ny)
mx, my = np.meshgrid(x, y)
z = np.outer(gaussian(x, nx / 2, nx / 2), gaussian(y, ny / 2, ny / 2))
inv_m = np.matrix([[i ** j for j in range(4)] for i in range(4)]) ** -1


def nearest(q, ksize):
    now = q * ksize
    now_q = np.int(now)
    if (now - now_q) < 0.5:
        return now_q
    else:
        return now_q + 1


def nearestNeighbor():
    arrz = []
    for i in range(knx):
        now_x = nearest(i, ksizex)
        arr = []
        for j in range(kny):
            now_y = nearest(j, ksizey)
            arr.append(z[now_x, now_y])
        arrz.append(arr)

    return np.array(arrz)


def linear(q, v1, v2):
    return v1 + q * (v2 - v1)


def bilinear():
    arrz = []
    pad_z = np.pad(z, ((0, 1), (0, 1)), 'edge')
    for i in range(knx):
        arr = []
        now_x = i * ksizex
        int_x = np.int(now_x)
        for j in range(kny):
            now_y = j * ksizey
            int_y = np.int(now_y)
            arr.append(linear(
                now_x - int_x,
                linear(now_y - int_y,
                       pad_z[int_x,     int_y],
                       pad_z[int_x,     int_y + 1]),
                linear(now_y - int_y,
                       pad_z[int_x + 1, int_y],
                       pad_z[int_x + 1, int_y + 1])))
        arrz.append(arr)
    return np.array(arrz)


def bicubic():
    arrz = []
    pad_z = np.pad(z, ((1, 3), (1, 3)), 'edge')
    for i in range(knx):
        arr = []
        now_x = i * ksizex + 1
        int_x = np.int(now_x)
        for j in range(knx):
            now_y = j * ksizey + 1
            int_y = np.int(now_y)

            x4 = np.array([(now_x - int_x + 1) ** k for k in range(4)])
            y4 = np.array([(now_y - int_y + 1) ** k for k in range(4)])
            y_arr = y4 * inv_m * pad_z[int_x - 1: int_x + 3,
                                       int_y - 1: int_y + 3].T
            ans = x4 * inv_m * y_arr.T
            arr.append(ans[0, 0])
        arrz.append(arr)
    return np.array(arrz)


def transform():
    data = np.pad(z, ((0, 1), (0, 1)), 'constant')
    y, x = np.meshgrid(np.arange(knx) * ksizey,
                       np.arange(kny) * ksizex)
    int_x = np.array(x, dtype=np.int32)
    int_y = np.array(y, dtype=np.int32)

    # bilinear
    return linear(x - int_x,
                  linear(y - int_y,
                         data[int_x,     int_y],
                         data[int_x,     int_y + 1]),
                  linear(y - int_y,
                         data[int_x + 1, int_y],
                         data[int_x + 1, int_y + 1]))


def testTime():
    t = timeit.timeit(bicubic, number=10)
    print(f"Bicubic: {t}s")
    t = timeit.timeit(bilinear, number=10)
    print(f"Bilinear: {t}s")
    t = timeit.timeit(transform, number=10)
    print(f"Bilinear: {t}s")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mx, my, z, color="red")
# new_z = nearestNeighbor()
# new_z = bilinear()
# new_z = bicubic()
new_z = transform()
ax.plot_surface(*np.meshgrid(np.arange(knx) * ksizex, np.arange(kny) * ksizey), new_z, alpha=1)
plt.show()
