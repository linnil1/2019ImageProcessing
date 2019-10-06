from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtChart import *
from PyQt5.Qt import *
import matplotlib.pyplot as plt
import numpy as np
import hw1_np as hw1
import hw2_np as hw2
import utils
import time


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
        # setup layout: title
        self.title = title
        self.func = func
        self.child = child
        self.n_parent = n_parent
        layout = QBoxLayout(QBoxLayout.TopToBottom)
        text = QLabel(title)
        text.setAlignment(Qt.AlignHCenter)
        layout.addWidget(text)
        self.layout = layout
        self.parents = []
        self.childrens = []

    def alert(self, text):
        alert(self.title + ": " + text)

    def setParent(self, parents):
        if not self.n_parent:
            self.alert("Cannot not set parent")
            return
        self.parents = parents

    def addChildren(self, children):
        if not self.child:
            self.alert("Cannot not set children")
            return
        self.childrens.append(children)

    def getParentImg(self):
        return [parent.img for parent in self.parents]

    def update(self):
        for child in self.childrens:
            child.update()

    def remove(self):
        removeRecursive(self.layout)


class ImageWidget(CommadWidget):
    """
    Image Widget, store the whole layout and it's image widget
    """
    def __init__(self, *args, **kwargs):
        super(ImageWidget, self).__init__(*args, **kwargs)
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
        super(ImageWidget, self).update()
        windowTighten()


class ImageReading(ImageWidget):
    """
    The layout of readRGB, read64
    """
    def __init__(self, text, func, image_format):
        super(ImageReading, self).__init__(text, func, n_parent=0)
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
        super(ImageSimple, self).__init__(*args)

    def update(self):
        img = self.func(*self.getParentImg())
        self.showImage(img)


class ImageSpinbox(ImageWidget):
    """
    ImageSpinbox inherited from ImageWidget and has spinbox feature
    """
    def __init__(self, title, func,
                 input_type, input_range, input_default, n_parent=1):
        super(ImageSpinbox, self).__init__(title, func, n_parent)
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
        super(ImageText, self).__init__(title, func, n_parent)
        self.widget_input = QLineEdit()
        self.widget_input.setInputMask(input_mask)
        self.widget_input.setText(input_default)
        self.widget_input.editingFinished.connect(self.update)
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
        super(ImageHistogram, self).__init__(title, func, child=False)

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
    module = module_name(*args)
    if module.n_parent:
        parents = [m for m in now_modules if m.child]
        if len(parents) < module.n_parent:
            alert(f"Add images first."
                  f"{module.title} needs {module.n_parent} input")
            return

        # update the image and add to lists
        module.setParent(parents[-module.n_parent:])
        for p in module.parents:
            p.addChildren(module)
        module.update()
    now_modules.append(module)
    layout.addLayout(module.layout)


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


# Predefine modules
modules = [
]

# setup and run
setUpMenu()
window.show()
app.exec_()
