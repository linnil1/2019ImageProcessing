import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def q2_12():
    n = 2
    x, y = np.meshgrid(np.linspace(-n, n), np.linspace(-n, n))
    m = 255 * np.exp(-(x ** 2 + y ** 2))
    m = np.uint(m)

    for i in range(8):
        k = (2 ** i)
        new_m = m // k * k
        new_m = np.uint(new_m)
        plt.subplot(2, 4, i + 1)
        plt.title(f"k = {8 - i}")
        plt.axis('off')
        plt.imshow(new_m, cmap="gray")

    plt.show()


def q2_18():
    def drawLine(sy, sx, ey, ex):
        x = np.linspace(sx, ex)
        y = np.linspace(sy, ey)
        plt.plot(x, y, color="blue")

    def poltVis(vis_m, allow_policy):
        nx, ny = vis_m.shape
        plt.imshow(vis_m, cmap="gray")
        for i in range(nx):
            for j in range(ny):
                color = "black" if vis_m[i, j] else "white"
                plt.text(j, i, str(m[i, j]),
                         ha="center", va="center", color=color)

        for i in range(nx):
            for j in range(ny):
                if not vis_m[i, j]:
                    continue
                allow_policy(vis_m, i, j)

    def connect_4(vis_m, i, j):
        nx, ny = vis_m.shape
        if i != nx - 1 and vis_m[i + 1, j]:
                drawLine(i, j, i + 1, j)
        if j != ny - 1 and vis_m[i, j + 1]:
                drawLine(i, j, i, j + 1)

    def connect_8(vis_m, i, j):
        connect_4(vis_m, i, j)
        nx, ny = vis_m.shape
        # to right bottom
        if i != nx - 1 and j != ny - 1 and vis_m[i + 1, j + 1]:
                drawLine(i, j, i + 1, j + 1)
        # to left bottom
        if i != nx - 1 and j != 0 and vis_m[i + 1, j - 1]:
                drawLine(i, j, i + 1, j - 1)

    def connect_m(vis_m, i, j):
        connect_4(vis_m, i, j)
        nx, ny = vis_m.shape
        # to right bottom
        if i != nx - 1 and j != ny - 1 and vis_m[i + 1, j + 1] and \
                not (vis_m[i + 1, j] or vis_m[i, j + 1]):
            drawLine(i, j, i + 1, j + 1)
        # to left bottom
        if i != nx - 1 and j != 0 and vis_m[i + 1, j - 1] and \
                not (vis_m[i + 1, j] or vis_m[i, j - 1]):
            drawLine(i, j, i + 1, j - 1)

    def drawConnect(vis_m, i):
        n = 2  # hard-code
        plt.subplot(n, 3, 3 * i + 1)
        plt.title("4-connected")
        poltVis(vis_m, connect_4)
        plt.subplot(n, 3, 3 * i + 2)
        plt.title("8-connected")
        poltVis(vis_m, connect_8)
        plt.subplot(n, 3, 3 * i + 3)
        plt.title("m-connected")
        poltVis(vis_m, connect_m)

    m = np.matrix("""
    3 1 2 1;
    2 2 0 2;
    1 2 1 1;
    1 0 1 2
    """)

    vis_m = np.zeros(m.shape)
    vis_m[(m == 0) | (m == 1)] = 1
    drawConnect(vis_m, 0)

    vis_m = np.zeros(m.shape)
    vis_m[(m == 2) | (m == 1)] = 1
    drawConnect(vis_m, 1)

    plt.show()


def q3_12():
    matplotlib.rcParams['text.usetex']=True

    x = np.linspace(0, 1, 1000)
    pr = 2 - 2 * x
    plt.plot(x, pr, label="$P_r$")

    r = 2 * x - x ** 2
    z = np.sqrt(r)
    dx = x - [0, *x[:-1]]
    dz = z - [0, *z[:-1]]
    plt.plot(z, pr * dx / dz, label="$P_z$ by ($z=\sqrt{2r - r^2}$)")
    plt.legend()
    plt.show()


q3_12()
