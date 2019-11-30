import numpy as np
import matplotlib.pyplot as plt
import argparse
import copy
import re
from functools import wraps


class Command():
    """
    Each command is a small units in OrderAction
    """
    def __init__(self, func, dest, layer, value=None):
        self.func = func
        self.dest = dest
        self.prev = layer[0]
        self.output = layer[1]
        self.value = value

    def __repr__(self):
        return f"<{self.dest} {self.prev}-{self.output}>{self.value}"

    def run(self, args):
        return self.func(*args, *self.value)


class OrderAction(argparse.Action):
    """
    OrderAction is to store the order of command.
    store_true can be set by nargs = 0
    """
    def __init__(self, option_strings, dest, layer=(1, 0), func=None, **kwargs):
        self.command = Command(func, dest, layer)
        if not kwargs.get("help"):
            kwargs["help"] = getDoc(func)
        if kwargs.get("nargs") is None:
            kwargs["nargs"] = 1
        super(OrderAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if "order_command" not in namespace:
            setattr(namespace, "order_command", [])
        order_command = namespace.order_command
        c = copy.deepcopy(self.command)
        c.value = values
        order_command.append(c)
        setattr(namespace, "order_command", order_command)


def getDoc(func):
    """
    Get description of the function
    """
    if not func.__doc__:
        return ""
    return func.__doc__.split(".")[0].strip()


def orderRun(parser):
    """
    Run the command with specific order
    """
    # read parameter
    args = parser.parse_args()
    if "order_command" not in args:
        parser.print_help()
        return
    print(args.order_command)

    # start postfix
    stack = []
    for command in args.order_command:
        # postfix
        if command.prev > len(stack):
            raise argparse.ArgumentTypeError(
                    f"Add more image before {command.dest} operation")
        if command.prev:
            imgs = stack[-command.prev:]
        else:
            imgs = []
        now_img = command.run(imgs)

        # remove previous and create new one
        if command.output == 0:
            stack = stack[:-command.prev]
            stack.append(now_img)
        # create new one
        elif command.output == 1:
            stack.append(now_img)
        # pop
        elif command.output == -1:
            stack = stack[:-command.prev]
        # no effect
        elif command.output is None:
            pass
        else:  # TODO
            pass

    plt.show()


def parserAdd_general(parser):
    """
    Add parser with general command
    """
    parser.add_argument("--copy",       type=str, nargs=0,
                        func=copyImg,   layer=(1, 1),    action=OrderAction,
                        help="Copy the previous Image")
    parser.add_argument("--pop",        type=str, nargs=0,
                        func=pop,       layer=(1, -1),   action=OrderAction,
                        help="Reomve previous Image")
    parser.add_argument("--show",       type=str, nargs=0,
                        func=show,      layer=(1, None), action=OrderAction,
                        help="Display the image")
    parser.add_argument("--showcolor",  type=str, nargs=0,
                        func=showColor, layer=(1, None), action=OrderAction,
                        help="Display the color image")
    parser.add_argument("--showgray",   type=str, nargs=0,
                        func=showGray,  layer=(1, None), action=OrderAction,
                        help="Display the image in gray scale")
    parser.add_argument("--show-noasp", type=str, nargs=0,
                        func=showNoasp, layer=(1, None), action=OrderAction,
                        help="Display the image without aspect ratio")


def copyImg(img):
    """ Copy """
    return img.copy()


def pop(img):
    return None


def showHist(bar):
    """ Display histogram """
    plt.figure()
    plt.title("Histogram")
    plt.bar(np.arange(bar.size), height=bar)
    return None


def showColor(img):
    """ Display: color image """
    plt.figure()
    plt.imshow(img)


def showGray(img):
    """ Display: gray scale image """
    plt.figure()
    plt.imshow(img, cmap="gray")


def showNoasp(img):
    """ Display: show without aspect ratio """
    plt.figure()
    plt.imshow(img, cmap="gray", aspect="auto")


def show(img):
    """
    Dispaly.
    Auto show image selected by img shape
    """
    if len(img.shape) == 3:
        showColor(img)
    else:
        showGray(img)


def parseSize(res):
    """
    Parse size from string.
    The string should be like 123x123
    """
    if not re.match(r"^\s*\d+\s*x\s*\d+\s*$", res):
        raise ValueError("The value is not like this format 123x123")
    return np.array(res.split('x'), dtype=np.int)


def normalize(img):
    """ Contrain image value from 0 to 1 """
    return (img - img.min()) / (img.max() - img.min())


def normalizeWrap(func):
    """ Wrap the normalize as decoder """
    @wraps(func)
    def wrapFunc(*args, **kwargs):
        return normalize(func(*args, **kwargs))
    return wrapFunc


# Affine transform
def linear(q, v1, v2):
    """ Linear interpolation """
    if q.shape == v1.shape:
        return v1 + (q - q.astype(np.int)) * (v2 - v1)
    else:
        return v1 + (q - q.astype(np.int))[..., None] * (v2 - v1)


def transform(img, affine, new_shape=None):
    """ Affine Transform with bilinear """
    # get locations of all points in new image
    if not new_shape:
        new_shape = img.shape
    new_img = np.zeros(new_shape)
    y, x = np.meshgrid(np.arange(new_img.shape[1]),
                       np.arange(new_img.shape[0]))
    z = np.ones(new_img.shape[:2])
    xyz = np.stack([x, y, z], 2)

    # get new locations
    affine = np.array(affine ** -1)
    pos = xyz.dot(affine.T)

    # get nonzero
    avail = (0 <= pos[:, :, 0]) & (pos[:, :, 0] < img.shape[0]) & \
            (0 <= pos[:, :, 1]) & (pos[:, :, 1] < img.shape[1])
    pos_avail = pos[avail]

    # add padding that ensure not larger than border
    if len(img.shape) == 2:
        data = np.pad(img, ((1, 2), (1, 2)), 'constant')
    else:
        data = np.pad(img, ((1, 2), (1, 2), (0, 0)), 'constant')
    int_x = np.array(pos_avail[:, 0], dtype=np.int32) + 1
    int_y = np.array(pos_avail[:, 1], dtype=np.int32) + 1

    # bilinear
    new_img[avail] = linear(pos_avail[:, 0],
                            linear(pos_avail[:, 1],
                                   data[int_x,     int_y],
                                   data[int_x,     int_y + 1]),
                            linear(pos_avail[:, 1],
                                   data[int_x + 1, int_y],
                                   data[int_x + 1, int_y + 1]))
    return new_img


def rotate(th):
    th *= np.pi / 180
    m = np.matrix(np.zeros([3, 3]))
    m[2, 2] = 1
    m[0, 0] = m[1, 1] = np.cos(th)
    m[0, 1] = -np.sin(th)
    m[1, 0] = np.sin(th)
    return m


def setMetrix(*loc):
    m = np.matrix(np.zeros([3, 3]))
    m[0, 0] = m[1, 1] = m[2, 2] = 1

    def wrap(r=1):
        m[loc] = r
        return m
    return wrap


# basic
Base   = setMetrix(2, 2)
transX = setMetrix(0, 2)
transY = setMetrix(1, 2)
shearX = setMetrix(0, 1)
shearY = setMetrix(1, 0)
scaleX = setMetrix(0, 0)
scaleY = setMetrix(1, 1)
