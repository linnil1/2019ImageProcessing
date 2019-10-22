import numpy as np
import hw2_np as hw2
import matplotlib.pyplot as plt

# a = hw2.readRGB("data/Image 3-2.JPG")

def gaussian(x, mu=0, sig=1):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

n = 25
x = np.arange(n)
y = gaussian(x, mu=n//2, sig=n//2)

kn = 1000
kx = np.arange(kn) / (kn - 1) * (n - 1)
arr = np.sinc(np.repeat([kx], n, axis=0).T - np.arange(n)).dot(y)

plt.plot(x, y, '.')
plt.plot(kx, arr)
plt.show()
