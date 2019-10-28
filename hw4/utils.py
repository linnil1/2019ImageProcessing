import numpy as np
import matplotlib.pyplot as plt
import argparse
import copy
import re


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
