from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtChart import *
from PyQt5.Qt import *
import matplotlib.pyplot as plt
import numpy as np
import hw1_np as hw1
import utils


# Qt Meta
app = QApplication([])
window = QMainWindow()
widget = QWidget()
layout = QBoxLayout(QBoxLayout.LeftToRight)
layout.addSpacerItem(QSpacerItem(100, 100))
widget.setLayout(layout)
window.setCentralWidget(widget)

# meta
default_size = (600, 400)

# state
oriimg = np.zeros([64, 64])
widget_oriimg = widget_resimg = widget_hist = None
now_func = None


def setUpMenu(modes):
    """
    Setup Menubar
    `modes` is array that each contains it's name and class that inherited form `MyFunc`
    """
    menubar = QMenuBar()
    # Menu for App
    menu_file = menubar.addMenu("App")
    act_clear = menu_file.addAction("Clear")
    act_clear.triggered.connect(lambda: recursiveRemove(layout))
    act_exit = menu_file.addAction("Exit")
    act_exit.triggered.connect(lambda: app.quit())

    # Menu for choosing mode
    menu_view = menubar.addMenu("view")
    for (name, func) in modes:
        act_mode = menu_view.addAction(name)
        act_mode.triggered.connect(func.setUp())

    window.setMenuWidget(menubar)


def setUpImageLayout(widget_img, load_func=False):
    """
    Setup layout for image.
    `widget_img` is the qlabel for image to show.
    `load_func` is the function called after image loaded.
    If `load_func` set to false, you cannot load image by button.
    """
    # add image
    layout_in = QBoxLayout(QBoxLayout.TopToBottom)
    widget_img.setAlignment(Qt.AlignTop)
    layout_in.addWidget(widget_img)
    layout.addLayout(layout_in)

    if load_func:
        # Add button
        button = QPushButton("Load Image")
        button.clicked.connect(readFile(widget_img, load_func))
        layout_in.addWidget(button)
    else:
        # Add text
        text = QLabel("Result")
        text.setAlignment(Qt.AlignHCenter)
        layout_in.addWidget(text)

    # return the widget object
    return widget_img


def showImage(img, widget_img):
    """
    Show the image on `widget_img`.
    Called this function after `setUpImageLayout`
    """
    # RGB image
    # qimage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
    # Gray scale
    qimage = QImage(np.uint8(img * 255), img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
    widget_img.setPixmap(QPixmap(qimage).scaled(*default_size, Qt.KeepAspectRatio))


def setOriimg(img):
    """
    This function can be used in `setUpImageLayout`
    """
    global oriimg
    oriimg = img


def setUpHistogramLayout():
    """
    Setup histogram for image.
    `widget_hist` is the variable that histogram plot showed.
    """
    global widget_hist
    layout_in = QBoxLayout(QBoxLayout.TopToBottom)
    layout.addLayout(layout_in)

    # bar chart widget
    widget_hist = QChartView()
    layout_in.addWidget(widget_hist)

    # text for histogram
    text = QLabel("Histogram")
    text.setAlignment(Qt.AlignHCenter)
    layout_in.addWidget(text)


def showHist(data):
    """
    Show histogram(barchart) after input an array of data.
    Called after `setUpHistogramLayout`.
    """
    # data to series
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
    widget_hist.setChart(chart)
    widget_hist.setFixedSize(*default_size)


def getFile():
    """
    Start file browser for user to select image
    """
    fd = QFileDialog()
    fd.setFileMode(QFileDialog.AnyFile)
    fd.setNameFilter("64 formatted image (*.64)")
    if fd.exec_():
        return fd.selectedFiles()[0]


def readFile(widget_img, load_func):
    """
    Update and load image file after `getFile` called.
    """
    def wrap():
        # get filename and load
        filename = getFile()
        if not filename:
            return
        img = now_func.read64(filename)
        load_func(img)

        # update screen
        showImage(img, widget_img)
        now_func.updateCustom()

    return wrap


def recursiveRemove(layout):
    """
    Clear all layout
    """
    for i in reversed(range(layout.count())):
        lay = layout.itemAt(i)
        if lay.layout() is not None:
            recursiveRemove(lay.layout())
        elif lay.widget() is not None:
            lay.widget().setParent(None)
        else:
            layout.removeItem(lay)
            # not work
            # lay.spacerItem().setParent(None)


class MyFunc():
    """
    This is the class that my qt app will call it.
    Two methods in it is very important: `setUpCustom`, `updateCustom`
    """

    @classmethod
    def setUp(cls):
        """
        This method is for loading this class
        """
        def wrap():
            global now_func
            now_func = cls()
            now_func.setUpCustom()
        return wrap

    def read64(self, filename):
        """
        Read image
        """
        return utils.read64(filename)

    @classmethod
    def cleanUp(cls, func):
        """
        This decorator is very convenience for hw1
        """
        def wrap(self, *arg, **kwargs):
            # remove
            recursiveRemove(layout)
            window.resize(window.sizeHint())
            global widget_oriimg, widget_resimg
            widget_oriimg = widget_resimg = None

            # setup oriimg
            widget_oriimg = setUpImageLayout(QLabel(), setOriimg)
            showImage(oriimg, widget_oriimg)

            # setup by your custom function
            func(self, *arg, **kwargs)

            # setup resimg and histogram
            widget_resimg = setUpImageLayout(QLabel(), load_func=False)
            setUpHistogramLayout()

            # update
            self.updateCustom()
        return wrap

    def updateCustom(self, arg=None):
        """
        This will be called when anything changed
        """
        pass

    def setUpCustom(self, *_):
        """
        Setup your own layout
        """
        pass


# My Function
# All of it inherited from MyFunc
class ImageBasic(MyFunc):
    """
    Basic
    """
    @MyFunc.cleanUp
    def setUpCustom(self, *_):
        pass

    def updateCustom(self, arg=None):
        img = oriimg
        showImage(img, widget_resimg)
        showHist(utils.getHist(img))


class ImageAdd(MyFunc):
    """
    Add image by a number
    """
    @MyFunc.cleanUp
    def setUpCustom(self, *_):
        # prefix text
        text = QLabel("+")
        layout.addWidget(text)

        # input for number
        self.widget_input = QDoubleSpinBox()
        self.widget_input.setRange(-255, 255)
        self.widget_input.setValue(0)
        self.widget_input.valueChanged.connect(self.updateCustom)
        layout.addWidget(self.widget_input)

    def updateCustom(self, arg=None):
        # calcute
        if arg is None:
            arg = self.widget_input.value()
        img = hw1.imageAdd(oriimg, arg)

        # update
        showImage(img, widget_resimg)
        showHist(utils.getHist(img))


class ImageMult(MyFunc):
    """
    Multiply image by a number
    """
    @MyFunc.cleanUp
    def setUpCustom(self, *_):
        # prefix text
        text = QLabel("*")
        layout.addWidget(text)

        # input for number
        self.widget_input = QDoubleSpinBox()
        self.widget_input.setRange(0, 255)
        self.widget_input.setValue(1)
        self.widget_input.valueChanged.connect(self.updateCustom)
        layout.addWidget(self.widget_input)

    def updateCustom(self, arg=None):
        # calcute
        if arg is None:
            arg = self.widget_input.value()
        img = hw1.imageMult(oriimg, arg)

        # update
        showImage(img, widget_resimg)
        showHist(utils.getHist(img))


class ImageAvg(MyFunc):
    """
    Calculate average from two images
    """
    @MyFunc.cleanUp
    def setUpCustom(self, *_):
        # Add number input
        text = QLabel("<- avg ->")
        layout.addWidget(text)

        # another img
        self.img1 = np.zeros(oriimg.shape)
        self.widget_input = setUpImageLayout(QLabel(), load_func=self.setImg)
        showImage(self.img1, self.widget_input)

    def setImg(self, img):
        self.img1 = img

    def updateCustom(self, arg=None):
        img = hw1.imageAvg(oriimg, self.img1)
        showImage(img, widget_resimg)
        showHist(utils.getHist(img))


class ImageSpecial(MyFunc):
    """
    Special operation by hw1-2-4
    """
    @MyFunc.cleanUp
    def setUpCustom(self, *_):
        pass

    def updateCustom(self, arg=None):
        img = hw1.image_special_func(oriimg)
        showImage(img, widget_resimg)
        showHist(utils.getHist(img))


# test
# oriimg = utils.read64("../JET.64")
# ImageSpecial.setUp()()

# setup custom things
setUpMenu([("imageBasic", ImageBasic),
           ("imageAdd", ImageAdd),
           ("imageMult", ImageMult),
           ("imageAvg", ImageAvg),
           ("ImageSpecial", ImageSpecial)])

# run
window.show()
app.exec_()
