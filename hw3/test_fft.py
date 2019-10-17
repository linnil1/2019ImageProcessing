import numpy as np
import utils
import hw2_np as hw2
import matplotlib.pyplot as plt


# data
data = [1, 2, 4, 4]
data2d = np.array([
    [1, 2, 42, 3, 0],
    [1, 12, 4, 3, 0],
    [51, 2, 44, 3, 0],
    [1, 2, 4, 34, 0]])
# large image testing
# n = 128
# n2 = n * n
# data2d = np.arange(n2).reshape(n, n)


# normal ft
def normal_ft():
    print(np.fft.fft(data))
    n = len(data)
    j, i = np.meshgrid(np.arange(n), np.arange(n))
    ftdata = np.sum(data * np.exp(-np.pi * 2j * i * j / n), axis=1)
    invftdata = np.real(np.sum(ftdata * np.exp(np.pi * 2j * i * j / n), axis=1) / n)
    print(ftdata)
    print(invftdata)


# fft
def fft(data, n):
    if len(data) == 1:
        return data

    even = fft(data[::2], n)
    odd  = fft(data[1::2], n)
    n_fft = len(data)
    i = np.arange(n_fft / 2)
    w = np.exp(-np.pi * 2j * i / n)
    fftdata = np.concatenate([even + odd * w,
                              even - odd * w])
    return fftdata


# inverse fft
def ifft(data, n):
    if len(data) == 1:
        return data

    even = fft(data[::2], n)
    odd  = fft(data[1::2], n)
    n_fft = len(data)
    i = np.arange(n_fft / 2)
    w = np.exp(np.pi * 2j * i / n)
    fftdata = np.concatenate([even + odd * w,
                              even - odd * w])
    return fftdata


# fft 2D
def fft2d(data):
    data_fft_row = np.stack([fft(i, len(i)) for i in data])
    data_fft_col = np.stack([fft(i, len(i)) for i in data_fft_row.T]).T
    return data_fft_col


def ifft2d(data):
    data_fft_row = np.stack([ifft(i, len(i)) for i in data])
    data_fft_col = np.stack([ifft(i, len(i)) for i in data_fft_row.T]).T
    return data_fft_col


# add padding to 2 power
def padWith2Power(data):
    nx = 2 ** np.int(np.ceil(np.log2(data.shape[0])))
    ny = 2 ** np.int(np.ceil(np.log2(data.shape[1])))
    new_data = np.zeros([nx, ny])
    new_data[:data.shape[0], :data.shape[1]] = data
    return new_data


def testfft():
    real_image = hw2.readRGB("data/Image 3-3.jpg")
    gray_image = hw2.toGrayA(real_image)
    # data2d = padWith2Power(gray_image)
    data2d = gray_image
    plt.subplot(1, 3, 1)
    plt.imshow(gray_image, cmap="gray")

    # fft
    data2d[0::2, 1::2] *= -1
    data2d[1::2, 0::2] *= -1
    fftdata = np.fft.fft2(data2d)

    # show spectrum
    b = np.log(1 + np.abs(fftdata))
    plt.subplot(1, 3, 2)
    plt.imshow(b / b.max(), cmap="gray")

    # cutoff high frequency
    cx = data2d.shape[0] // 2
    cy = data2d.shape[1] // 2
    j, i = np.meshgrid(np.arange(data2d.shape[1]) - cy,
                       np.arange(data2d.shape[0]) - cx)
    fftdata[i ** 2 + j ** 2 > 100] = 0

    # inverse it
    post_data = np.real(np.fft.ifft2(fftdata))
    post_data = post_data[:gray_image.shape[0], :gray_image.shape[1]]
    post_data[0::2, 1::2] *= -1
    post_data[1::2, 0::2] *= -1

    # plot
    plt.subplot(1, 3, 3)
    plt.imshow(post_data, cmap="gray")
    plt.show()

def testConv():
    real_image = hw2.readRGB("data/Image 3-4.jpg")
    gray_image = hw2.toGrayA(real_image)
    # data2d = padWith2Power(gray_image)
    data2d = gray_image
    plt.subplot(1, 3, 1)
    plt.imshow(gray_image, cmap="gray")

    # fft
    fftdata1 = np.fft.fft2(data2d)
    f = np.zeros(fftdata1.shape)
    f[:3, :3] = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]])

    # show kernal spectrum
    fftdata2 = np.fft.fft2(f)
    b = np.log(1 + np.abs(fftdata2))
    cx = b.shape[0] // 2
    cy = b.shape[1] // 2
    b = np.concatenate([b[cx:,], b[:cx,]], axis=0)
    b = np.concatenate([b[:,cy:], b[:,:cy]], axis=1)
    plt.subplot(1, 3, 2)
    plt.imshow(b / b.max(), cmap="gray")

    
    # convolute and inverse it
    fftdata3 = fftdata1 * fftdata2
    post_data = np.real(np.fft.ifft2(fftdata3))
    post_data = post_data[:gray_image.shape[0], :gray_image.shape[1]]
    plt.subplot(1, 3, 3)
    plt.imshow(post_data, cmap="gray")
    plt.show()


# normal_ft()
# print(fft(data))

# print(fft2d(data2d))
print(np.fft.fft2(data2d))
# print(np.real(ifft2d(fft2d(data2d)) / data2d.size))

# import timeit
# print(timeit.timeit(lambda: np.fft.fft2(data2d), number=100))
# print(timeit.timeit(lambda: fft2d(data2d), number=10))
# testfft()
# testConv()
