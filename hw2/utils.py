import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt


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
    def __init__(self, option_strings, dest, nargs=1,
                 layer=(1, 0), func=None, **kwargs):
        self.nargs = nargs
        self.command = Command(func, dest, layer)
        super(OrderAction, self).__init__(option_strings, dest,
                                          nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if 'order_command' not in namespace:
            setattr(namespace, 'order_command', [])
        order_command = namespace.order_command
        c = copy.deepcopy(self.command)
        c.value = values
        order_command.append(c)
        setattr(namespace, 'order_command', order_command)


def parserAdd_general(parser):
    """
    Add parser with general command
    """
    parser.add_argument('--copy',     type=str, nargs=0, help="Copy the previous Image",
                        func=copyImg,  layer=(1, 1),    action=OrderAction)
    parser.add_argument('--pop',      type=str, nargs=0, help="Reomve previous Image",
                        func=pop,      layer=(1, -1),   action=OrderAction)
    parser.add_argument('--show',     type=str, nargs=0, help="Display the image",
                        func=show,     layer=(1, None), action=OrderAction)
    parser.add_argument('--showgray', type=str, nargs=0, help="Display the image in gray scale",
                        func=showGray, layer=(1, None), action=OrderAction)


def copyImg(img):
    """
    Copy the image
    """
    return img.copy()


def pop(img):
    return None


def showHist(bar):
    """
    Show histogram
    """
    plt.figure()
    plt.title("Histogram")
    plt.bar(np.arange(bar.size), height=bar)
    return None


def show(img):
    """
    Show color image
    """
    plt.figure()
    plt.imshow(img)


def showGray(img):
    """
    Show image in gray scale
    """
    plt.figure()
    plt.imshow(img, cmap="gray")


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
