import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui


class HoverableCurveItem(pg.PlotCurveItem):
    sigCurveHovered = QtCore.Signal(object, object)
    sigCurveNotHovered = QtCore.Signal(object, object)

    def __init__(self, hoverable=True, *args, **kwargs):
        super(HoverableCurveItem, self).__init__(*args, **kwargs)
        self.hoverable = hoverable
        self.setAcceptHoverEvents(True)

    def hoverEvent(self, ev):
        if self.hoverable:
            if self.mouseShape().contains(ev.pos()):
                self.sigCurveHovered.emit(self, ev)
            else:
                self.sigCurveNotHovered.emit(self, ev)


class MainWindow(QtGui.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.view = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.view)
        self.makeplot()

    def makeplot(self):
        x = [1, 2, 3, 4, 5, 6, 5, 1, 1]
        y = [0, 1, 2, 3, 3, 4, 6, 7, 0]
        plot = self.view.addPlot()
        self.plotitem = HoverableCurveItem(x, y, pen=pg.mkPen('w', width=10))
        self.plotitem.setClickable(True, width=150)
        self.plotitem.sigCurveHovered.connect(self.hovered)
        self.plotitem.sigCurveNotHovered.connect(self.leaveHovered)
        plot.addItem(self.plotitem)

    def hovered(self):
        print("cursor entered curve")
        self.plotitem.setPen(pg.mkPen('b', width=10))

    def leaveHovered(self):
        self.plotitem.setPen(pg.mkPen('w', width=10))


if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())