import sys
import time
from PyQt5 import QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np


class ExPlot(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        desktop = QtWidgets.QDesktopWidget()
        width = desktop.screenGeometry().width()
        ratio = width / 1920
        self.resize(1400 * ratio, 800 * ratio)

        self.layout = QtWidgets.QHBoxLayout(self)
        self.plotWidget = pg.GraphicsLayoutWidget(self)
        self.layout.addWidget(self.plotWidget)
        self.plot = self.plotWidget.addPlot(1, 1, enableMenu=False)

        self.indexes = []
        self.counter = 1
        self.time = 0
        self.x = np.arange(0, 10, 0.01)
        self.y = np.sin(self.x)

        plotData = self.plot.plot(self.x, self.y)
        self.plotWidget.scene().sigMouseClicked.connect(self.getIndex)
        self.points_plot = self.plot.plot(x=[], y=[], pen=None, symbol='o')
        self.points_plot.sigPointsClicked.connect(self.removePoint)

    def getIndex(self, event):
        if event.button() == 2:
            items = self.plotWidget.scene().items(event.scenePos())
            for item in items:
                if isinstance(item, pg.ViewBox):
                    index = int(item.mapSceneToView(event.scenePos()).x() / 0.01)
                    if 0 <= index < len(self.x):
                        self.addPoint(index)

    def addPoint(self, index):
        if index not in self.indexes:
            self.indexes.append(index)
            x = self.x[self.indexes]
            y = self.y[self.indexes]
            self.points_plot.setData(x, y, pen=None)

    def removePoint(self, item, points):
        tdiff = time.time() - self.time
        if tdiff < 1:
            point_x = points[0].pos()[0]
            ix = int(round(point_x / 0.01))
            if ix in self.indexes:
                self.indexes.remove(ix)
                x = self.x[self.indexes]
                y = self.y[self.indexes]
                self.points_plot.setData(x, y, pen=None)

        self.time = time.time()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    # app = QtGui.QApplication(sys.argv)
    ex = ExPlot()
    ex.show()
    sys.exit(app.exec_())