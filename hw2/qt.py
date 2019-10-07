from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtChart import *
from PyQt5.Qt import *
import matplotlib.pyplot as plt
import numpy as np
import hw1_np as hw1
import hw2_np as hw2
import utils


# Window instance
app = QApplication([])
window = QMainWindow()
widget = QWidget()
layout = QBoxLayout(QBoxLayout.LeftToRight)
widget.setLayout(layout)
window.setCentralWidget(widget)

# instance
now_modules = []

# meta
default_size = (600, 400)


class CommadWidget():
    """
    Command Widget, a basic widget for image and histogram
    """
    def __init__(self, title, func, n_parent=1, child=True):
        self.title = title
        self.func = func
        self.child = child
        self.n_parent = n_parent
        self.index = np.inf

        # setup layout and title
        layout = QBoxLayout(QBoxLayout.TopToBottom)
        # setup title
        self.title_widget = QLabel(title)
        self.title_widget.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self.title_widget)
        self.layout = layout

        # upstream and downstream
        self.parents = []
        self.childrens = []

    def __repr__(self):
        return "<{} {}>({}, {})".format(self.index, self.title,
                                        [p.index for p in self.parents],
                                        [c.index for c in self.childrens])

    def getParentImg(self):
        """
        Get image from upstream(parent)
        """
        return [parent.img for parent in self.parents]

    def update(self):
        """
        Update the downstream
        """
        for child in self.childrens:
            child.update()

    def remove(self):
        """
        Remove this module
        """
        removeRecursive(self.layout)

    def alert(self, text):
        alert(self.title + ": " + text)

    def setParent(self):
        """
        Update parents (also it's childrens) when something changed
        """
        # remove old
        for p in self.parents:
            p.childrens.remove(self)

        # Update Parents
        self.parents = []
        for dropdown in self.dropdowns:
            module = dropdown.itemData(dropdown.currentIndex())
            self.parents.append(module)

        # Update childrens
        for p in self.parents:
            if not p.child:
                self.alert("Cannot not set children")
                return
            p.childrens.append(self)
        self.update()

    def setupWithAll(self, modules):
        """
        Setup the dropdown after your class is finished
        NOTE: modules is same as now_modules
        """
        # set index
        self.index = len(modules)
        self.title_widget.setText(f"[{self.index}] {self.title}")

        # check need to set parent
        if not self.n_parent:
            return

        # find available parents
        options = [m for m in modules if m.child]

        # check legal or not
        if len(options) < self.n_parent:
            str_err = f"Add images first."\
                      f"{self.title} needs {self.n_parent} input"
            raise SyntaxError(str_err)

        # Set text description
        layout = QBoxLayout(QBoxLayout.LeftToRight)
        text = QLabel("From: ")
        layout.addWidget(text)

        # set widget
        self.dropdowns = []
        for i in range(self.n_parent):
            dropdown = QComboBox()
            for opt in options:
                dropdown.addItem(str(opt.index), opt)
            dropdown.setCurrentText(str(options[-self.n_parent + i].index))
            dropdown.currentIndexChanged.connect(self.setParent)
            layout.addWidget(dropdown)
            self.dropdowns.append(dropdown)
        self.layout.addLayout(layout)

        # update the image and add to lists
        self.setParent()


class ImageWidget(CommadWidget):
    """
    Image Widget, store the whole layout and it's image widget
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # setup image widget
        self.widget = QLabel()
        self.layout.addWidget(self.widget)
        self.layout.setAlignment(Qt.AlignTop)

    def showImage(self, img):
        """
        This method will find out the display method by your input
        """
        self.img = img
        if len(img.shape) == 3:
            qimage = QImage(np.uint8(img * 255), img.shape[1], img.shape[0],
                            img.shape[1] * img.shape[2], QImage.Format_RGB888)
        else:
            qimage = QImage(np.uint8(img * 255), img.shape[1], img.shape[0],
                            img.shape[1], QImage.Format_Grayscale8)
        self.widget.setPixmap(QPixmap(qimage).scaled(
            *default_size, Qt.KeepAspectRatio))
        super().update()
        windowTighten()


class ImageReading(ImageWidget):
    """
    The layout of readRGB, read64
    """
    def __init__(self, text, func, image_format):
        super().__init__(text, func, n_parent=0)
        self.image_format = image_format

        # load image button
        button = QPushButton("Load Image")
        button.clicked.connect(self.load)
        self.layout.addWidget(button)

        # show blank image
        self.showImage(np.zeros([100, 100]))

    def load(self):
        """
        Read file and update
        """
        filename = self.getFile()
        if not filename:
            return
        img = self.func(filename)
        self.showImage(img)

    def getFile(self):
        """
        Start file browser for user to select image
        """
        fd = QFileDialog()
        fd.setFileMode(QFileDialog.AnyFile)
        fd.setNameFilter(self.image_format)
        if fd.exec_():
            return fd.selectedFiles()[0]


class ImageSimple(ImageWidget):
    """
    Image Widget, store the whole layout and it's image widget
    """
    def __init__(self, *args):
        super().__init__(*args)

    def update(self):
        img = self.func(*self.getParentImg())
        self.showImage(img)


class ImageSpinbox(ImageWidget):
    """
    ImageSpinbox inherited from ImageWidget and has spinbox feature
    """
    def __init__(self, title, func,
                 input_type, input_range, input_default, n_parent=1):
        super().__init__(title, func, n_parent)
        if input_type is float:
            self.widget_input = QDoubleSpinBox()
        elif input_type is int:
            self.widget_input = QSpinBox()
        self.widget_input.setRange(*input_range)
        self.widget_input.setValue(input_default)
        self.widget_input.valueChanged.connect(self.update)
        self.layout.addWidget(self.widget_input)

    def update(self, arg=None):
        if arg is None:
            arg = self.widget_input.value()
        img = self.func(*self.getParentImg(), arg)
        self.showImage(img)


class ImageText(ImageWidget):
    """
    ImageSpinbox inherited from ImageWidget and has spinbox feature
    """
    def __init__(self, title, func,
                 input_default="", input_mask="", n_parent=1):
        super().__init__(title, func, n_parent)
        self.widget_input = QLineEdit()
        self.widget_input.setInputMask(input_mask)
        self.widget_input.setText(input_default)
        self.widget_input.returnPressed.connect(self.update)
        self.layout.addWidget(self.widget_input)

    def update(self, arg=None):
        if arg is None:
            arg = self.widget_input.displayText()
        img = self.func(*self.getParentImg(), arg)
        self.showImage(img)


class ImageHistogram(CommadWidget):
    """
    The layout of histogram
    """
    def __init__(self, title, func):
        super().__init__(title, func, child=False)

        # setup bar chart widget
        self.widget = QChartView()
        self.layout.addWidget(self.widget)

    def update(self):
        """
        Get bin from image
        Note: It's is not recommand to use too many bins
        """
        data = self.func(*self.getParentImg())

        # bin data to bar series
        barset = QBarSet("Ratio of occurrences")
        for i in data:
            barset.append(i)
        barseries = QBarSeries()
        barseries.append(barset)

        # chart related
        chart = QChart()
        chart.addSeries(barseries)
        # chart.setAnimationOptions(QChart.SeriesAnimations)

        # show axis(the code order is important)
        axisY = QValueAxis()
        chart.addAxis(axisY, Qt.AlignLeft)
        chart.legend().setVisible(False)
        # axisY.setRange(0, 1)
        barseries.attachAxis(axisY)

        # add to screen
        self.widget.setChart(chart)
        self.widget.setFixedSize(*default_size)


# Functions for existing modules list
def layoutRemoveAll():
    """
    Clear all layout
    """
    for i in reversed(range(len(now_modules))):
        moduleRemoveLast()


def moduleRemoveLast():
    """
    Remove the last one
    """
    if len(now_modules) == 0:
        alert("Cannot remove. It's empty")
        return
    now_modules.pop().remove()
    windowTighten()


def moduleAdd(module_name, args):
    """
    Add new module
    """
    # print(module_name, args)
    try:
        module = module_name(*args)
        module.setupWithAll(now_modules)
        now_modules.append(module)
        layout.addLayout(module.layout)
    except Exception as e:
        alert(e)


# Setup
def setUpMenu():
    """
    Setup Menubar
    Modules should be set in global
    """
    menubar = QMenuBar()
    # Menu for App
    menu_app = menubar.addMenu("App")
    act_exit = menu_app.addAction("Exit")
    act_exit.triggered.connect(lambda: app.quit())

    # Menu for Action
    menu_action = menubar.addMenu("Action")
    act_clearall = menu_action.addAction("Clear All")
    act_clearall.triggered.connect(lambda: layoutRemoveAll())
    act_back = menu_action.addAction("BackSpace")
    act_back.triggered.connect(lambda: moduleRemoveLast())

    # Menu for adding module
    menu_modules = menubar.addMenu("Modules")
    for i in range(len(modules)):
        act_mode = menu_modules.addAction(modules[i][1][0])
        # Be carefor to python lambda function
        act_mode.triggered.connect(
                (lambda m: lambda: moduleAdd(*m))(modules[i]))

    window.setMenuWidget(menubar)


# help function
def windowTighten():
    """
    Tighten the window
    Not work everytime. wwwww
    """
    widget.resize(widget.sizeHint())
    window.resize(window.sizeHint())


def removeRecursive(layout):
    """
    Remove all things in specific layout
    Does it have memory leak?
    """
    layout.setParent(None)
    for i in reversed(range(layout.count())):
        lay = layout.itemAt(i)
        if lay.widget():
            lay.widget().setParent(None)
        elif lay.layout():
            removeRecursive(lay.layout())


def alert(error):
    """
    Send a alert
    """
    message = QMessageBox(QMessageBox.Icon.Critical, "Error", str(error))
    message.exec()


def test():
    filename = "data/kemono_friends.jpg"
    img = hw2.readRGB(filename)
    moduleAdd(*modules[1])
    now_modules[0].showImage(img)
    filename = "data/stackoverflow.jpg"
    img = hw2.readRGB(filename)
    moduleAdd(*modules[1])
    now_modules[1].showImage(img)
    moduleAdd(*modules[2])
    moduleAdd(*modules[-2])


# Predefine modules
modules = [
    # read image module
    (ImageReading,   ("Read 64 formatted image",  hw1.read64, "64 formatted image (*.64)")),
    (ImageReading,   ("Read color image",         hw2.readRGB, "JPG or BMP image (*.jpg *.jpeg *.bmp)")),
    # histogram module
    (ImageHistogram, ("Histogram (32bin)",        hw1.getHist)),
    # hw1 module
    (ImageSpinbox,   ("Add number to image",      hw1.imageAdd, float, (-255, 255), 0)),
    (ImageSpinbox,   ("Multiply number to image", hw1.imageMult, float, (0, 255), 1)),
    (ImageSimple,    ("Difference between image", hw1.imageDiff, 2)),
    (ImageSimple,    ("Average between image",    hw1.imageAvg, 2)),
    (ImageSimple,    ("Special function in hw1",  hw1.image_special_func)),
    # hw2 module
    (ImageSimple,    ("Gray scale (A)",           hw2.toGrayA)),
    (ImageSimple,    ("Gray scale (B)",           hw2.toGrayB)),
    (ImageSpinbox,   ("Set threshold",            hw2.setThreshold, int, (0, 255), 128)),
    (ImageSimple,    ("Histogram equalization",   hw2.histogramEqualize)),
    (ImageSpinbox,   ("Gamma Correction",         hw2.gammaCorrection, float, (-100, 100), 1)),
    (ImageText,      ("Resize (Bilinear)",        hw2.resizeFromStr, "600x400", "")),
    # util module
    (ImageSimple,    ("Copy",                     utils.copyImg)),
]

# setup and run
setUpMenu()
window.show()
# test()
app.exec_()
