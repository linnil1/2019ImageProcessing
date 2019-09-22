from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtChart import *
from PyQt5.Qt import *
import matplotlib.pyplot as plt
import numpy as np

app = QApplication([])
widget = QWidget()
parent_layout = QBoxLayout(QBoxLayout.TopToBottom)
modes = ["123", "abc"]


def applyMode(index):
    print(index)
    print(modes[index])
    for i in reversed(range(layout.count())):
        layout_in = layout.itemAt(i).layout()
        for l in reversed(range(layout_in.count())):
            layout_in.itemAt(l).widget().setParent(None)
        layout.itemAt(i).layout().setParent(None)
    img = plt.imread("tmp.jpg")
    img = np.uint8(img[:,:,:3]).copy()
    showImage(layout, img)
    showHist(layout, [i * 0.03 for i in range(32)])


# Bar for choosing mode
def setUp_choosing():
    layout = QBoxLayout(QBoxLayout.RightToLeft)
    modebox = QComboBox()
    for mode in modes:
        modebox.addItem(mode)
    modebox.currentIndexChanged.connect(applyMode)
    layout.addWidget(modebox)
    layout.addWidget(QLabel("Choose mode: "))
    layout.setContentsMargins(0, 0, 0, 30)
    parent_layout.addLayout(layout)


# Main layout
## Original image
def showImage(layout, img):
    layout_in = QBoxLayout(QBoxLayout.TopToBottom)
    qimage = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
    qimage_display = QLabel()
    qimage_display.setPixmap(QPixmap(qimage))
    layout_in.addWidget(qimage_display)

    # load image button
    load_button = QPushButton("Load Image")
    load_button.clicked.connect(getFile)
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
    layout_in.addWidget(chartview)
    layout.addLayout(layout_in)

    text = QLabel("Histogram")
    text.setAlignment(Qt.AlignHCenter)
    layout_in.addWidget(text) 


# Load file with dialog
def getFile():
    fd = QFileDialog()
    fd.setFileMode(QFileDialog.AnyFile)
    fd.setNameFilter("64 formatted image (*.64)")
    if fd.exec_():
        return fd.selectedFiles()[0]


setUp_choosing()
layout = QBoxLayout(QBoxLayout.LeftToRight)
img = plt.imread("tmp.jpg")
img = np.uint8(img[:,:,:3]).copy()
showImage(layout, img)
showHist(layout, [i * 0.03 for i in range(32)])
parent_layout.addLayout(layout)
widget.setLayout(parent_layout)
# widget.resize(1200, 800)
widget.show()
app.exec_()
