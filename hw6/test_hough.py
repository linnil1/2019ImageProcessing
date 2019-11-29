import numpy as np
import matplotlib.pyplot as plt
import hw3_np as hw3
from sklearn import cluster, mixture


img = plt.imread("data/part3/rects.bmp")[:, :] / 255
edge = np.abs(hw3.LoGConv(img, sig=3)) > .5
edge = edge[30:, 30:]
max_dis = np.ceil(np.sqrt(np.sum(np.array(edge.shape) ** 2))).astype(np.int) + 1

# hough
th = np.linspace(0, np.pi, 181)
x, y = np.where(edge)
r = (np.outer(x, np.cos(th)) + np.outer(y, np.sin(th))).astype(np.int)
hough = np.zeros([2 * max_dis, th.size])
for i in range(len(x)):
    hough[r[i] + max_dis, np.arange(th.size)] += 1

# k-mean
k = 8
crit = 70
hough_crit = hough.copy()
hough_crit[hough < crit] = 0
x, y = np.where(hough_crit)
model = cluster.KMeans(k).fit(np.array([y, x]).T, sample_weight=hough_crit[x, y])
centers = model.cluster_centers_
centers = centers[centers.argsort(axis=0)[:, 0]]
print(centers)

"""
centers = [
    [  21.46595995,  817.20094229],
    [  20.69287289, 1081.7038942 ],
    [  49.98192989, 1156.67437658],
    [  50.02811562,  914.84441636],
    [ 105.26107446,  798.21677663],
    [ 108.29040657,  671.29641498],
    [ 130.16452809,  897.08511548],
    [ 131.4446664 ,  647.58409908],
]
"""

# reverse
x = np.arange(0, edge.shape[0])
restore_edge = np.zeros(edge.shape)
for i, center in enumerate(centers):
    th = center[0] / 180 * np.pi
    y = ((center[1] - max_dis - x * np.cos(th)) / np.sin(th)).astype(np.int)
    index_in = np.logical_and(y >= 0, y < edge.shape[1])
    restore_edge[x[index_in], y[index_in]] = (i // 2 + 1) / 4

# plot
plt.figure(figsize=(10, 8))
plt.subplot(221)
plt.title("Original Image")
plt.imshow(img, cmap="gray")

plt.subplot(222)
plt.title("Edge Detection")
plt.imshow(edge, cmap="gray")

plt.subplot(223)
plt.title("Hough transform")
plt.imshow(hough, cmap="gray", aspect="auto")
for center in model.cluster_centers_:
    c = plt.Circle(center, 10, color="r", fill=False)
    plt.gca().add_artist(c)
plt.gca().invert_yaxis()

plt.subplot(224)
plt.title("Restore the edge")
plt.imshow(restore_edge)
plt.show()
