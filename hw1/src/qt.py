from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtChart import *
from PyQt5.Qt import *
import matplotlib.pyplot as plt
import numpy as np

app = QApplication([])
window = QMainWindow()
widget = QWidget()
layout = QBoxLayout(QBoxLayout.LeftToRight)
modes = ["123", "abc"]


def setUpMenu():
    menu = QMenuBar()
    menu_all = menu.addMenu("File")
    act_exit = menu_all.addAction("Exit")
    act_exit.triggered.connect(lambda : app.quit())

    menu_view = menu.addMenu("view")
    for mode in modes:
        act_mode = menu_view.addAction(mode)
        act_mode.triggered.connect(applyMode)

    window.setMenuWidget(menu)


def applyMode():
    for i in reversed(range(layout.count())):
        print(layout.itemAt(i))
        layout.removeItem(layout.itemAt(i))
    img = plt.imread("tmp.jpg")
    img = np.uint8(img[:,:,:3]).copy()
    showImage(layout, img)
    # showHist(layout, [i * 0.03 for i in range(32)])


# Main layout
## Original image
def showImage(layout, img):
    # image
    layout_in = QBoxLayout(QBoxLayout.TopToBottom)
    qimage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
    qimage_display = QLabel()
    qimage_display.setPixmap(QPixmap(qimage))

    # load image button
    load_button = QPushButton("Load Image")
    load_button.clicked.connect(getFile)

    layout_in.addWidget(qimage_display)
    layout_in.addWidget(load_button)
    layout.addLayout(layout_in)


## histogram
def showHist(layout, data):
    layout_in = QBoxLayout(QBoxLayout.TopToBottom)
    bar_data= QBarSet("123")
    for i in data:
        bar_data.append(i)
    bars = QBarSeries()
    bars.append(bar_data)
    chart = QChart()
    chart.addSeries(bars)
    chart.setAnimationOptions(QChart.SeriesAnimations)
    axisY = QValueAxis()
    axisY.setRange(0, 1)
    chart.addAxis(axisY, Qt.AlignLeft)
    bars.attachAxis(axisY);
    chartview = QChartView(chart)
    chartview.setFixedSize(layout.itemAt(0).layout().itemAt(0).widget().sizeHint())

    text = QLabel("Histogram")
    text.setAlignment(Qt.AlignHCenter)

    layout_in.addWidget(chartview)
    layout_in.addWidget(text) 
    layout.addLayout(layout_in)


# Load file with dialog
def getFile():
    fd = QFileDialog()
    fd.setFileMode(QFileDialog.AnyFile)
    fd.setNameFilter("64 formatted image (*.64)")
    if fd.exec_():
        return fd.selectedFiles()[0]


# my
def getValue(*arg):
    print(arg)

def custom_setup(layout):
    text = QLabel("+")
    input_float = QDoubleSpinBox()
    input_float.valueChanged.connect(getValue)
    layout.addWidget(text)
    layout.addWidget(input_float)


setUpMenu()
img = plt.imread("tmp.jpg")
img = np.uint8(img[:,:,:3]).copy()
# showImage(layout, img)
custom_setup(layout)
# showHist(layout, [i * 0.03 for i in range(32)])
widget.setLayout(layout)
# widget.resize(1200, 800)
window.setCentralWidget(widget)
window.show()
app.exec_()
