import matplotlib.pyplot as plt
import numpy as np


def gaussian(x, mu=0, sig=1):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


n = 25
kn = 200
ksize = (n - 1) / (kn - 1)
x = np.arange(n)
y = gaussian(x, mu=n//2, sig=n//2)
y[::2] -= 0.05


def nearestNeighbor():
    arr = []
    for i in range(kn):
        now = i * ksize
        now_x = np.int(now)
        if (now - now_x) < 0.5:
            arr.append(y[now_x])
        else:
            arr.append(y[now_x + 1])
    return arr


def linear():
    arr = []
    pad_y = np.concatenate([y, [y[-1]]])
    for i in range(kn):
        now = i * ksize
        now_x = np.int(now)
        b, a = y[now_x], pad_y[now_x + 1] - y[now_x]
        arr.append(b + a * (now - now_x))
    return arr


def cubic():
    arr = []
    pad_y = np.concatenate([[y[0]], y, [y[-1]] * 4])
    inv_m = np.matrix([[i ** j for j in range(4)] for i in range(4)]) ** -1
    print(inv_m)
    for i in range(kn):
        now = i * ksize + 1
        now_x = np.int(now)
        # set 0 is at now_x -1
        in_arr = np.array([(now - now_x + 1) ** j for j in range(4)])
        ans = (in_arr * inv_m).dot(pad_y[now_x - 1: now_x + 3])[0, 0]
        arr.append(ans)
    return arr


plt.plot(x, y, "o")
# new_y = nearestNeighbor()
# new_y = linear()
new_y = cubic()
plt.plot(np.arange(kn) * ksize, new_y)
plt.show()
