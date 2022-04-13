__all__ = ['Viewer']

import pyqtgraph as pg
import sys
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
# from PyQt5 import QtCore, QtGui
import numpy as np
import cv2

class Graph(pg.GraphItem):
    def __init__(self):
        self.dragPoint = None
        self.dragOffset = None
        self.downPos = None
        self.textItems = []
        pg.GraphItem.__init__(self)
        self.scatter.sigClicked.connect(self.clicked)

    def setData(self, **kwds):
        self.text = kwds.pop('text', [])
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.setTexts(self.text)
        self.updateGraph()

    def setTexts(self, text):
        for i in self.textItems:
            i.scene().removeItem(i)
        self.textItems = []
        for t in text:
            item = pg.TextItem(t)
            self.textItems.append(item)
            item.setParentItem(self)

    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        for i, item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])

    def mouseDragEvent(self, ev):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ShiftModifier:
            shift = True
        else:
            shift = False

        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return

        if ev.isStart():

            # We are already one step into the drag.
            # Find the point(s) at the mouse cursor when the button was first
            # pressed:
            self.downPos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(self.downPos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.dragOffset = self.data['pos'][ind] - self.downPos
            self.dragOffsets = []
            self.startPoints = []
            for pnt in self.data['pos']:
                self.dragOffsets.append(pnt - self.downPos)
                self.startPoints.append(pnt)

        elif ev.isFinish():
            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return

        if shift:
            # move all the points
            ind = self.dragPoint.data()[0]
            n_points = len(self.data['pos'])
            if ind == 0 or ind == n_points-1:
                # rotate about the end point
                dragAmount = ev.pos()[1] + self.dragOffsets[ind][1] - self.startPoints[ind][1]
                for i, (pnt, s_pnt) in enumerate(zip(self.data['pos'], self.startPoints)):
                    if ind == 0:
                        pnt[1] = self.startPoints[i][1] + dragAmount * (n_points-i)/n_points
                    else:
                        pnt[1] = self.startPoints[i][1] + dragAmount * i/n_points


            else:
                # move all the same
                for i, pnt in enumerate(self.data['pos']):
                    pnt[1] = ev.pos()[1] + self.dragOffsets[i][1]
        else:
            # move one point only
            ind = self.dragPoint.data()[0]
            self.data['pos'][ind][1] = ev.pos()[1] + self.dragOffsets[ind][1]

        self.updateGraph()
        ev.accept()

    def clicked(self, pts):
        print("clicked: %s" % pts)

class Widget(QtWidgets.QWidget):
    def __init__(self):
        super(Widget, self).__init__()
        self.init_ui()
        self.qt_connections()

        image = np.random.normal(size=(30, 60, 3), loc=1024, scale=64).astype(np.uint16)
        self.set_image(image)

    def init_ui(self):
        self.setWindowTitle('Viewer')
        self.widget = pg.GraphicsLayoutWidget()
        self.img = pg.ImageItem(border='w')
        self.g = Graph()
        view = self.widget.addViewBox()
        view.addItem(self.img)
        view.addItem(self.g)

        self.increasebutton = QtWidgets.QPushButton("Increase Amplitude")
        self.decreasebutton = QtWidgets.QPushButton("Decrease Amplitude")

        horizontalLayout = QtWidgets.QHBoxLayout()
        horizontalLayout.addWidget(self.increasebutton)
        horizontalLayout.addWidget(self.decreasebutton)

        verticalLayout = QtWidgets.QVBoxLayout()
        verticalLayout.addWidget(self.widget)

        verticalLayout.addLayout(horizontalLayout)
        self.setLayout(verticalLayout)

        # self.setGeometry(10, 10, 1000, 600)
        self.show()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            print('bye')
            self.deleteLater()
        elif event.key() == QtCore.Qt.Key_Enter:
            print('self.proceed()')
        event.accept()


    def qt_connections(self):
        self.increasebutton.clicked.connect(self.on_increasebutton_clicked)
        self.decreasebutton.clicked.connect(self.on_decreasebutton_clicked)

    def set_image(self, image):
        out = cv2.transpose(image)
        out = cv2.flip(out, flipCode=1)
        self.img.setImage(out)
        # self.img.setImage(image)
        n_points = 20
        (c,r,z) = out.shape
        x = np.linspace(1, c, n_points+1)
        pos = np.column_stack((x, np.ones(n_points+1) * r//4))
        adj = np.stack([np.arange(n_points), np.arange(1, n_points+1)]).transpose()

        pen = pg.mkPen('r', width=2)
        # self.g.setData(pos=pos, size=10, pen=pen, symbol='o', symbolPen=pen, symbolBrush='r', pxMode=True)
        self.g.setData(pos=pos, adj=adj, size=20, pen=pen, symbol='o', symbolPen=pen, symbolBrush=(50, 50, 200, 00), pxMode=True)

    def on_increasebutton_clicked(self):
        print ("Amplitude increased")

    def on_decreasebutton_clicked(self):
        print ("Amplitude decreased")


import cv2

class Viewer:
    def __init__(self, process):
        self.timer = QtCore.QTimer()
        self.timer.setSingleShot(True)
        self.process = process

        app = QtWidgets.QApplication(sys.argv)
        app.setApplicationName('Viewer')
        self.imageView = Widget()

        self.timer.timeout.connect(self.run_timer)
        self.timer.start(1)
        self.app = app.exec_()


    def close(self):
        sys.exit(self.app)

    def run_timer(self):
        go = self.process(self)
        if go:
            self.timer.start(1)

    def update_image(self, image):
        self.imageView.set_image(image)



if __name__ == '__main__':

    def test(_viewer):
        image = cv2.imread('/home/jn/data/Karioitahi_09Feb2022/132MSDCF-28mm-f4-small/DSC01013.JPG')
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('test', image)
        k = cv2.waitKey(1)
        if k == ord('g'):
            print(k)
        if k == ord('u'):
            _viewer.update_image(image_rgb)
        if k == ord('q') or k == 27:
            return False

        return True

    viewer = Viewer(test)
    viewer.close()

