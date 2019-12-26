import numpy as np
import matplotlib.pyplot as plt

i = 0
imgs = []
skes = []

num = np.random.randint(0, 50, size=10, replace=False)
for i in num:
    imgs.append(plt.imread(f"result/20/{i}.B.jpg"))
    skes.append(plt.imread(f"result/20/{i}.BA.jpg"))

img = np.concatenate(imgs, axis=1)
ske = np.concatenate(skes, axis=1)

a = np.concatenate([img, ske], axis=0)

plt.imsave("pack_1.jpg", a)
plt.imshow(a)
plt.show()
