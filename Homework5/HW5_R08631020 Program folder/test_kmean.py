import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import cluster, mixture
import scipy
import hw2_np as hw2
import hw5_np as hw5


def kMean(data, k):
    epoch = 10
    eps = 1e-6
    ans = np.zeros([k, 2])
    ans = data[np.random.choice(np.arange(n * 4), k, replace=False), :]
    for i in range(epoch):
        dist = np.sum((data - ans[:, None, :]) ** 2, axis=2)
        closest = np.argmin(dist, axis=0)
        now_ans = np.array([np.mean(data[closest == i], axis=0) for i in range(k)])
        if np.sum(np.abs(ans - now_ans)) < eps:
            break
        ans = now_ans
    return ans


def show_test():
    n = 200
    data = np.zeros([n * 4, 2])
    np.random.seed(123)

    data[:n * 2, 0] = np.random.normal(0, 3, size=n * 2)
    data[:n * 2, 1] = np.random.normal(2, 1, size=n * 2)
    data[n * 2:n * 3, 0] = np.random.normal(12, 1, size=n)
    data[n * 2:n * 3, 1] = np.random.normal(4, 1, size=n)
    data[n * 3:, 0] = np.random.normal(6, 2, size=n)
    data[n * 3:, 1] = np.random.normal(10, 2, size=n)

    # real
    plt.subplot(121)
    plt.title("Dataset")
    plt.plot(data[:n * 2,   0], data[:n * 2,   1], 'r.')
    plt.plot(data[n * 2:n * 3, 0], data[n * 2:n * 3, 1], 'g.')
    plt.plot(data[n * 3:,   0], data[n * 3:,   1], 'b.')

    k = 3
    # model = cluster.KMeans(k).fit(data)
    model = cluster.SpectralClustering(k).fit(data)
    group = model.labels_

    """
    # my own kmean
    ans = kMean(data, k)
    dist = np.sum((data - ans[:, None, :]) ** 2, axis=2)
    group = np.argmin(dist, axis=0)
    print(ans)
    """

    plt.subplot(122)
    plt.title("Predict by kmean")
    # ans = model.cluster_centers_
    # plt.plot(ans[:,  0], ans[:, 1], 'k.')
    for i in range(k):
        plt.plot(data[group == i, 0], data[group == i, 1], color[i] + '.')
    plt.show()


# settings
k = 6
space = hw5.Color.RGB
img = hw2.readRGB("data/HW05-3-03.bmp")
size = (68 * 3, 102 * 3)
title = f"K-means(k={k}) on {space.value}"

# Read the image, resize, filter, and transformation
img = hw5.fromRGB(img, space)
img_resize = hw2.bilinear(img, size)
data = img_resize.reshape(img_resize.shape[0] * img_resize.shape[1], 3)
data = data[:, :3]

# models
# model = mixture.GaussianMixture(k).fit_predict(data)
# group = model
# model = cluster.SpectralClustering(k).fit(data)
model = cluster.KMeans(k).fit(data)
group = model.labels_

# set color
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = [int(i[1:], 16) for i in colors]
colors = np.array([[i // 256 // 256, i // 256 % 256, i % 256] for i in colors])

# plot
fig = plt.figure(figsize=(20, 16))

plt.tight_layout(pad=0, rect=0)
ax = fig.add_subplot(221, projection='3d')
ax.scatter(*np.transpose(data, (1, 0)))
ax.set_xlabel(space.value[0])
ax.set_ylabel(space.value[1])
ax.set_zlabel(space.value[2])
ax.view_init(30, 75)

ax = fig.add_subplot(222, projection='3d')
for i in range(k):
    ax.scatter(*np.transpose(data[group == i], (1, 0)))
ax.set_title(title)
ax.set_xlabel(space.value[0])
ax.set_ylabel(space.value[1])
ax.set_zlabel(space.value[2])
ax.view_init(30, 75)

"""
# show in 2D
plt.subplot(221)
plt.plot(*np.transpose(data, (1, 0)), '.')
plt.xlabel("H")
plt.ylabel("S")
plt.subplot(222)
plt.title(title)
for i in range(k):
    plt.plot(*np.transpose(data[group==i], (1, 0)), '.')
plt.xlabel("H")
plt.ylabel("S")
"""

# show image
plt.subplot(223)
img_resize = hw5.toRGB(img_resize, space)
plt.imshow(img_resize)
plt.subplot(224)
plt.title(title)
new_img = colors[group % 10]
new_img = new_img.reshape(img_resize.shape)
plt.imshow(new_img)

plt.show()
