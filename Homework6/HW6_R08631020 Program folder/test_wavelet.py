import numpy as np
import matplotlib.pyplot as plt
import utils


# harr basic
harr_scale = np.array([1, 1]) / np.sqrt(2)
harr_wavel = np.array([1, -1]) / np.sqrt(2)
harr_scale_r = np.flip(harr_scale)
harr_wavel_r = np.flip(harr_wavel)


def wavelet1D(data):
    data = np.array([1, 4, -3, 0])
    n2 = 2

    # wavelet
    result = []
    for i in range(n2):
        result.append(np.convolve(data, harr_wavel_r)[1::2])
        data = np.convolve(data, harr_scale_r)[1::2]
    result.append(data)
    result = np.concatenate(list(reversed(result)))
    print(result)

    # inverse wavelet
    data = result
    data_ori = data[0]
    data = data[1:]
    for i in range(n2):
        wavel = np.zeros(2 ** (i + 1))
        wavel[0::2] = data[:2 ** i]
        scale = np.zeros(2 ** (i + 1))
        scale[0::2] = data_ori
        data_ori = np.convolve(wavel, harr_wavel)[:-1] + \
                   np.convolve(scale, harr_scale)[:-1]
        data = data[2 ** i:]

    print(data_ori)


def padTo2(img):
    s = np.array(2 ** np.ceil(np.log2(img.shape)), dtype=np.int)
    a = (s - img.shape) // 2
    return np.pad(img, list(zip(a, (s - img.shape) - a)))


def convolve(data, f):
    return np.stack([np.convolve(d, f) for d in data])


def wavelet2D(data, depth):
    if not depth:
        return data
    # by column
    scale = convolve(data.T, harr_scale_r)[:, 1::2].T
    wavel = convolve(data.T, harr_wavel_r)[:, 1::2].T

    # by row
    scale_scale = convolve(scale, harr_scale_r)[:, 1::2]
    wavel_h     = convolve(scale, harr_wavel_r)[:, 1::2]
    wavel_v     = convolve(wavel, harr_scale_r)[:, 1::2]
    wavel_wavel = convolve(wavel, harr_wavel_r)[:, 1::2]

    # recursion
    scale_scale = wavelet2D(scale_scale, depth - 1)
    return np.vstack([np.hstack([scale_scale, wavel_h]),
                      np.hstack([wavel_v, wavel_wavel])])


def upSample(data):
    z = np.zeros([data.shape[0], data.shape[1] * 2])
    z[:, ::2] = data
    return z


def wavelet2DInv(data, depth):
    if not depth:
        return data
    h, w = np.array(data.shape) // 2
    # recursion
    scale_scale = wavelet2DInv(data[:h, :w], depth - 1)

    # by row
    scale_scale = convolve(upSample( scale_scale), harr_scale)[:, :-1]
    wave_h      = convolve(upSample(data[:h, w:]), harr_wavel)[:, :-1]
    wave_v      = convolve(upSample(data[h:, :w]), harr_scale)[:, :-1]
    wavel_wavel = convolve(upSample(data[h:, w:]), harr_wavel)[:, :-1]

    # by column
    scale = convolve(upSample((scale_scale + wave_h).T), harr_scale)[:, :-1].T
    wavel = convolve(upSample((wavel_wavel + wave_v).T), harr_wavel)[:, :-1].T
    return wavel + scale


img = plt.imread("data/part2/set1/clock1.JPG")[:, :, 0] / 255
img = img[:256, :256]
data = padTo2(img)
print(data.shape)

n = data.shape[0]
n2 = 2

# wavelet
wave_img = wavelet2D(data, 1)
print(wave_img.shape)
# wave_img[:128, :128] = 0
restore_img = wavelet2DInv(wave_img, 1)
print(restore_img.shape)

# plt.imshow(wave_img, cmap="gray")
plt.subplot(121)
plt.title("Original img")
plt.imshow(data, cmap="gray")
plt.subplot(122)
plt.title("Wavelet transform image highpass")
# plt.imshow(restore_img, cmap="gray")
plt.imshow(wave_img, cmap="gray")
plt.show()

data = img
n = 4
n2 = 2
