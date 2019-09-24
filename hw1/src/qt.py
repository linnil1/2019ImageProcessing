from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtChart import *
from PyQt5.Qt import *
import matplotlib.pyplot as plt
import numpy as np
import hw1_np
import utils


app = QApplication([])
window = QMainWindow()
widget = QWidget()
layout = QBoxLayout(QBoxLayout.LeftToRight)
layout.addSpacerItem(QSpacerItem(100, 100))
widget.setLayout(layout)
# widget.resize(1200, 800)
window.setCentralWidget(widget)
default_size = (600, 400)

ori_img = np.zeros([64, 64])
widget_oriimg = widget_resimg = widget_hist = None
now_func = None


def setUpMenu(modes):
    menu = QMenuBar()
    menu_all = menu.addMenu("File")
    act_exit = menu_all.addAction("Exit")
    act_exit.triggered.connect(lambda : app.quit())

    menu_view = menu.addMenu("view")
    for (name, func) in modes:
        act_mode = menu_view.addAction(name)
        act_mode.triggered.connect(func.setUp())

    window.setMenuWidget(menu)


# Main layout
## Original image
def setUpImageLayout(widget_img, load=False):
    layout_in = QBoxLayout(QBoxLayout.TopToBottom)
    widget_img.setAlignment(Qt.AlignTop)
    layout_in.addWidget(widget_img)
    layout.addLayout(layout_in)

    # load image button
    if load:
        load_button = QPushButton("Load Image")
        load_button.clicked.connect(readFile(widget_img, load))
        layout_in.addWidget(load_button)
    else:
        text = QLabel("Result")
        text.setAlignment(Qt.AlignHCenter)
        layout_in.addWidget(text)

    return widget_img


def showImage(img, widget_img):
    # image
    # qimage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
    qimage = QImage(np.uint8(img * 255), img.shape[1], img.shape[0], QImage.Format_Grayscale8)
    widget_img.setPixmap(QPixmap(qimage).scaled(*default_size, Qt.KeepAspectRatio))


def setOriimg(img):
    global ori_img
    ori_img = img


## histogram
def setUpHistogram():
    global widget_hist
    widget_hist = chartview = QChartView()

    text = QLabel("Histogram")
    text.setAlignment(Qt.AlignHCenter)

    layout_in = QBoxLayout(QBoxLayout.TopToBottom)
    layout_in.addWidget(chartview)
    layout_in.addWidget(text) 
    layout.addLayout(layout_in)


def showHist(data):
    layout_in = QBoxLayout(QBoxLayout.TopToBottom)
    bar_data= QBarSet("")
    for i in data:
        bar_data.append(i)
    bars = QBarSeries()
    bars.append(bar_data)
    chart = QChart()
    chart.addSeries(bars)
    # chart.setAnimationOptions(QChart.SeriesAnimations)
    axisY = QValueAxis()
    # axisY.setRange(0, 1)
    chart.addAxis(axisY, Qt.AlignLeft)
    bars.attachAxis(axisY);
    chartview = widget_hist
    chartview.setChart(chart)
    chartview.setFixedSize(*default_size)


# Load file with dialog
def getFile():
    fd = QFileDialog()
    fd.setFileMode(QFileDialog.AnyFile)
    fd.setNameFilter("64 formatted image (*.64)")
    if fd.exec_():
        return fd.selectedFiles()[0]


def readFile(widget_img, load_func):
    def wrap():
        filename = getFile()
        if not filename:
            return
        img = now_func.read64(filename)
        load_func(img)
        showImage(img, widget_img)
        now_func.updateCustom()
    return wrap


def recursiveRemove(layout):
    for i in reversed(range(layout.count())):
        print(layout.itemAt(i))
        # layout.removeItem(layout.itemAt(i))
        lay = layout.itemAt(i)
        if lay.layout() is not None:
            recursiveRemove(lay.layout())
        elif lay.widget() is not None:
            lay.widget().setParent(None)
        else:
            layout.removeItem(lay)
            # lay.spacerItem().setParent(None)


class MyFunc():
    @classmethod
    def setUp(cls):
        def wrap():
            global now_func
            now_func = cls()
            now_func.setUpCustom()
        return wrap

    def read64(self, filename):
        return utils.read64(filename)

    @classmethod
    def cleanUp(cls, func):
        def wrap(self, *arg, **kwargs):
            recursiveRemove(layout)

            global widget_oriimg, widget_resimg
            widget_oriimg = widget_resimg = None

            widget_oriimg = setUpImageLayout(QLabel(), setOriimg)
            showImage(ori_img, widget_oriimg)
            func(self, *arg, **kwargs)
            setUpHistogram()
            showHist(utils.getHist(ori_img))
        return wrap


    def updateCustom(self, arg=None):
        pass

    def setUpCustom(self, *_):
        pass

    def setUpResimg(self):
        global widget_resimg
        widget_resimg = setUpImageLayout(QLabel(), load=False)
        showImage(ori_img, widget_resimg)


# my
class ImageAdd(MyFunc):
    def updateCustom(self, arg=None):
        if arg == None:
            arg = self.widget_input.value()
        img = hw1_np.imageAdd(ori_img, arg)
        showImage(img, widget_resimg)
        showHist(utils.getHist(img))

    @MyFunc.cleanUp
    def setUpCustom(self, *_):
        # Add number input
        text = QLabel("+")
        input_float = QDoubleSpinBox()
        input_float.setRange(-255, 255)
        input_float.setValue(0)
        input_float.valueChanged.connect(self.updateCustom)
        self.widget_input = input_float
        layout.addWidget(text)
        layout.addWidget(input_float)
        self.setUpResimg()


# ori_img = utils.read64("../JET.64")
setUpMenu([("imageAdd", ImageAdd),
           ("imageAvg", ImageAvg)])
ImageAdd.setUp()()
# showImage(layout, img)
# custom_setup(layout)
# showHist(layout, [i * 0.03 for i in range(32)])
window.show()
app.exec_()
