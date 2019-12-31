import numpy as np
import matplotlib.pyplot as plt

i = 0
imgs = []
skes = []

num = np.random.choice(range(20), size=10, replace=False)
for i in num:
    imgs.append(plt.imread(f"result/neural_style/task{i}_ori.png"))
    skes.append(plt.imread(f"result/neural_style/task{i}_style.png"))

img = np.concatenate(imgs, axis=1)
ske = np.concatenate(skes, axis=1)

a = np.concatenate([img, ske], axis=0)

plt.imsave("pack_3.jpg", a)
plt.imshow(a)
plt.show()
