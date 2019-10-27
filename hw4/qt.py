from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtChart import *
from PyQt5.Qt import *
import matplotlib.pyplot as plt
import numpy as np
import hw1_np as hw1
import hw2_np as hw2
import hw3_np as hw3
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
    def __init__(self, func, n_parent=1, child=True):
        self.func = func
        self.title = utils.getDoc(func)
        self.child = child
        self.n_parent = n_parent
        self.index = np.inf

        # setup layout and title
        layout = QBoxLayout(QBoxLayout.TopToBottom)
        # setup title
        self.title_widget = QLabel(self.title)
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
        Update the objcet and downstream
        """
        # update this object with error handling
        try:
            self.updateThis()
        except Exception as e:
            self.alert(e)
            self.setBlank()

        # update downstream
        for child in self.childrens:
            child.update()

    def updateThis(self):
        """
        The function used for updating the instance
        """
        pass

    def remove(self):
        """
        Remove this module
        """
        removeRecursive(self.layout)

    def alert(self, text):
        alert(self.title + ": " + str(text))

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
            self.update()
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

    def showImage(self):
        """
        This method will find out the display method by your input
        """
        img = self.img.copy()
        if len(img.shape) == 3:
            qimage = QImage(np.uint8(img * 255), img.shape[1], img.shape[0],
                            img.shape[1] * img.shape[2], QImage.Format_RGB888)
        else:
            qimage = QImage(np.uint8(img * 255), img.shape[1], img.shape[0],
                            img.shape[1], QImage.Format_Grayscale8)
        self.widget.setPixmap(QPixmap(qimage).scaled(
            *default_size, Qt.KeepAspectRatio))

    def update(self, *arg, **kwargs):
        super().update()
        self.showImage()
        windowTighten()

    def setBlank(self):
        self.img = np.zeros(default_size)


class ImageReading(ImageWidget):
    """
    The layout of readRGB, read64
    """
    def __init__(self, func, image_format):
        super().__init__(func, n_parent=0)
        self.image_format = image_format

        # load image button
        button = QPushButton("Load Image")
        button.clicked.connect(self.load)
        self.layout.addWidget(button)

        # show blank image
        self.setBlank()

    def load(self):
        """
        Read file and update
        """
        filename = self.getFile()
        if not filename:
            return
        self.img = self.func(filename)
        self.update()

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

    def updateThis(self):
        self.img = self.func(*self.getParentImg())


class ImageSpinbox(ImageWidget):
    """
    ImageSpinbox inherited from ImageWidget and has spinbox feature.
    The input_arr contains a list of tuple that given
    type, range, default value, name.
    """
    def __init__(self, func, input_arr, n_parent=1):
        super().__init__(func, n_parent)
        self.widget_inputs = []
        layout = QBoxLayout(QBoxLayout.LeftToRight)
        for i in input_arr:
            input_type, input_range, input_default = i[:3]
            if len(i) >= 4:
                layout.addWidget(QLabel(i[3]))
            if input_type is float:
                widget_input = QDoubleSpinBox()
            elif input_type is int:
                widget_input = QSpinBox()
            widget_input.setRange(*input_range)
            widget_input.setValue(input_default)
            widget_input.valueChanged.connect(self.update)
            self.widget_inputs.append(widget_input)
            layout.addWidget(widget_input)
        self.layout.addLayout(layout)

    def updateThis(self):
        args = []
        for widget_input in self.widget_inputs:
            args.append(widget_input.value())
        self.img = self.func(*self.getParentImg(), *args)


class ImageText(ImageWidget):
    """
    ImageSpinbox inherited from ImageWidget and has spinbox feature
    """
    def __init__(self, func,
                 input_default="", input_mask="", n_parent=1):
        super().__init__(func, n_parent)
        self.widget_input = QLineEdit()
        self.widget_input.setInputMask(input_mask)
        self.widget_input.setText(input_default)
        self.widget_input.returnPressed.connect(self.update)
        self.layout.addWidget(self.widget_input)

    def updateThis(self):
        arg = self.widget_input.displayText()
        self.img = self.func(*self.getParentImg(), arg)


class ImageHistogram(CommadWidget):
    """
    The layout of histogram
    """
    def __init__(self, func):
        super().__init__(func, child=False)

        # setup bar chart widget
        self.widget = QChartView()
        self.layout.addWidget(self.widget)

    def updateThis(self):
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
        act_mode = menu_modules.addAction(utils.getDoc(modules[i][1]))
        # Be carefor to python lambda function
        act_mode.triggered.connect(
                (lambda m: lambda: moduleAdd(m[0], m[1:]))(modules[i]))

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
    filename = "data/Image 3-4.jpg"
    # filename = "data/stackoverflow.jpg"
    img = hw2.readRGB(filename)
    moduleAdd(*modules[1])
    now_modules[0].img = img
    now_modules[0].update()
    moduleAdd(*modules[9])
    moduleAdd(*modules[-1])


# Predefine modules
modules = [
    # read image module
    (ImageReading, hw1.read64,           "64 formatted image (*.64)"),
    (ImageReading, hw2.readRGB,          "JPG or BMP image (*.JPG *.JPEG *.jpg *.jpeg *.bmp)"),
    # histogram module
    (ImageHistogram, hw1.getHist),
    # util module
    (ImageSimple,  utils.copyImg),
    # hw1 module
    (ImageSpinbox, hw1.imageAdd,         [(float, (-255, 255), 0)]),
    (ImageSpinbox, hw1.imageMult,        [(float, (0, 255), 1)]),
    (ImageSimple,  hw1.imageDiff, 2),
    (ImageSimple,  hw1.imageAvg, 2),
    (ImageSimple,  hw1.image_special_func),
    # hw2 module
    (ImageSimple,  hw2.toGrayA),
    (ImageSimple,  hw2.toGrayB),
    (ImageSpinbox, hw2.setThreshold,     [(int, (0, 255), 128)]),
    (ImageSimple,  hw2.histogramEqualize),
    (ImageSpinbox, hw2.gammaCorrection,  [(float, (-100, 100), 1)]),
    (ImageText,    hw2.bilinear,         "600x400", ""),
    # hw3 module
    (ImageText,    hw3.medianFilter,     "3x3", ""),
    (ImageText,    hw3.minFilter,        "3x3", ""),
    (ImageText,    hw3.maxFilter,        "3x3", ""),
    (ImageText,    hw3.boxFilter,        "3x3", ""),
    (ImageSpinbox, hw3.idealLowpass,     [(float, (0, 1000), 20, "cutoff")]),
    (ImageSpinbox, hw3.gaussian,         [(float, (0, 1000), 20, "cutoff")]),
    (ImageSpinbox, hw3.butterworth,      [(float, (1, 1000), 20, "cutoff"),
                                          (float, (1, 1000), 1, "n")]),
    (ImageSpinbox, hw3.unsharp,          [(float, (1, 1000), 2, "k"),
                                          (float, (1, 1000), 20, "cutoff")]),
    (ImageSpinbox, hw3.sobelH,           [(float, (1, 1000), 2)]),
    (ImageSpinbox, hw3.sobelV,           [(float, (1, 1000), 2)]),
    (ImageSpinbox, hw3.roberGx,          [(float, (1, 1000), 2)]),
    (ImageSpinbox, hw3.roberGy,          [(float, (1, 1000), 2)]),
    (ImageSpinbox, hw3.laplacian4,       [(float, (1, 1000), 2)]),
    (ImageSpinbox, hw3.laplacian8,       [(float, (1, 1000), 2)]),
    (ImageText,    hw3.customKernal,     "0 0 0; 0 1 0; 0 0 0"),
    (ImageSpinbox, hw3.LoG,              [(float, (1, 1000), 2, "k"),
                                          (float, (1, 100), 1, "sigma")]),
]

# setup and run
setUpMenu()
window.show()
# test()
app.exec_()
