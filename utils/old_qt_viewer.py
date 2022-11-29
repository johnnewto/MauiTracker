__all__ = ['Viewer']

import pyqtgraph as pg
import sys

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import QPushButton
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
# from PyQt5 import QtCore, QtGui
import numpy as np
from utils.image_utils import *
from utils.g_images import *
from utils import parameters as pms
from utils.horizon import find_sky_2
import cv2
import csv
from pathlib import Path
from utils.graph import Graph

import logging
logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=pms.LOGGING_LEVEL)
logger = logging.getLogger(__name__)


class MyPushButton(QPushButton):
    def __init__(self, *args):
        QPushButton.__init__(self, *args)

    def event(self, event):
        if (event.type() == QEvent.KeyPress) and (event.key() == Qt.Key_Space):
            event.ignore()
            return True

        return QPushButton.event(self, event)


class DrawingImage(pg.ImageItem):

    def __init__(self, parent):
        super().__init__()
        self.parent: Widget = parent
        # self.setAutoDownsample(False)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            if self.raiseContextMenu(ev):
                ev.accept()

    def raiseContextMenu(self, ev):
        self.downPos = ev.pos()
        menu = self.getMenu()
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(pos.x(), pos.y()))
        return True

    def getMenu(self):
        """
        Create the menu
        """
        if self.menu is None:
            self.menu = QtGui.QMenu()
            self._addregion = QtGui.QAction("Add Region", self.menu)
            self._addregion.triggered.connect(self.graph_addregion)
            self.menu.addAction(self._addregion)

        return self.menu

    def graph_addregion(self):
        print("graph_addregion", self.downPos)
        self.parent.set_graph_points(pos, name="RECT")
        # self.parent.graphs.append()
        # data = self.data
        # pos = list(data['pos'])
        # data['pos'] = np.append(data['pos'], [[128,128]], axis=0)
        # data['adj'][-1] = np.array([3,4])
        # data['adj'] = np.append(data['adj'], [[4,0]], axis=0)
        pass



class Widget(QtWidgets.QWidget):
    def __init__(self):
        super(Widget, self).__init__()
        self.init_ui()
        self.qt_connections()
        self._key_press = None
        self.viewCurrentImage()

    def init_ui(self):
        self.setWindowTitle('Viewer')
        self.widget = pg.GraphicsLayoutWidget()
        # self.img = pg.ImageItem(border='w')
        self.img = DrawingImage(self)
        self.graphs = []
        # self.graphs = [Graph(name='Horizon')]
        # self.graphnames = ['obj1']
        self.view = self.widget.addViewBox()
        self.view.addItem(self.img)
        # view.addItem(self.graphs[0])

        self.btn_autoLabel = MyPushButton("Auto Horizon")
        self.btn_saveCSV = MyPushButton("Save CSV")
        self.btn_readCSV = MyPushButton("Read CSV")
        self.btn_free3 = MyPushButton("Free")
        self.cbx_saveTrainMask = QtWidgets.QCheckBox("Save Training Mask")

        horizontalLayout1 = QtWidgets.QHBoxLayout()
        horizontalLayout1.addWidget(self.btn_autoLabel)
        horizontalLayout1.addWidget(self.btn_saveCSV)
        horizontalLayout1.addWidget(self.btn_readCSV)
        horizontalLayout1.addWidget(self.btn_free3)
        horizontalLayout1.addWidget(self.cbx_saveTrainMask)

        verticalLayout = QtWidgets.QVBoxLayout()
        verticalLayout.addWidget(self.widget)

        verticalLayout.addLayout(horizontalLayout1)
        self.setLayout(verticalLayout)
        self.setGeometry(100, 100, 1500, 1000)
        self.show()



    def setDataChanged(self, _bool, name=None):
        self.setGraphColour(_bool, name)
        # do this after changing the colour
        for graph in self.graphs:
            if name is None or graph.name == name:
                graph.dataChanged = _bool

    def getDataChanged(self, name=None):
        for graph in self.graphs:
            if graph.name == name:
                return self.graphs[0].dataChanged
        else:
            return False

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q or event.key() == QtCore.Qt.Key_Escape :
            self.deleteLater()

        self._key_press = event.key()
        event.accept()

    def qt_get_keypress(self):
        ret = self._key_press
        self._key_press = None
        return ret

    def qt_connections(self):
        self.btn_autoLabel.clicked.connect(self.btn_autoLabel_clicked)
        self.btn_saveCSV.clicked.connect(self.btn_saveCSV_clicked)
        self.btn_readCSV.clicked.connect(self.btn_readCSV_clicked)
        self.btn_free3.clicked.connect(self.btn_free3button_clicked)

    def viewCurrentImage(self, img=None):
        try:
            if img is None:
                img = getGImages().small_rgb
            out = cv2.transpose(img)
            out = cv2.flip(out, flipCode=1)
            self.img.setImage(out)
            self.display_image = out


        except Exception as e:
            logger.error(e)
            pass

    # def new_graph(self, pos=None, name=None):
    #     if pos is None:
    #         return
    #     else:
    #         n_points = pos.shape[0]
    #         (cols, rows) = self.display_image.shape[:2]
    #         pos[:,0] = pos[:,0] * cols
    #         pos[:,1] = pos[:,1] * rows
    #         assert max(pos[:,0]) < cols , 'columns must be less than image width '
    #     adj = np.stack([np.arange(n_points), np.arange(1, n_points+1)]).transpose()
    #     adj[-1,1] = 0
    #     graph = Graph(self, name=name)
    #     # graph.setClickable(True, width=150)
    #     self.graphs.append(graph)
    #     self.view.addItem(self.graphs[-1])
    #     self.graphs[-1].setData(pos=pos, adj=adj, size=10, symbol='o', symbolBrush=(50, 50, 200, 00), pxMode=False)
    #     return graph
    #
    #
    # def set_graph_points(self, pos=None, name=None):
    #     if pos is None:
    #         n_points = pms.NUM_HORIZON_POINTS
    #         (c,r) = self.display_image.shape[:2]
    #         x = np.linspace(0, c-1, n_points)
    #         pos = np.column_stack((x, np.ones(n_points) * r//4))
    #         pass
    #     else:
    #         n_points = pos.shape[0]
    #         (cols, rows) = self.display_image.shape[:2]
    #         pos[:,0] = pos[:,0] * cols
    #         pos[:,1] = pos[:,1] * rows
    #         assert max(pos[:,0]) < cols , 'columns must be less than image width '
    #     adj = np.stack([np.arange(n_points), np.arange(1, n_points+1)]).transpose()
    #     adj[-1,1] = 0
    #     # pen = pg.mkPen('g', width=2)
    #     # self.graph.setData(pos=pos, adj=adj, size=10, pen=pen, symbol='o', symbolPen=pen, symbolBrush=(50, 50, 200, 00), pxMode=False)
    #     done = False
    #     for graph in self.graphs:
    #         if graph.name == name:
    #             graph.setData(pos=pos, adj=adj, size=10, symbol='o', symbolBrush=(50, 50, 200, 00), pxMode=False)
    #             done = True
    #     if not done:
    #         self.graphs.append(Graph(self, name=name))
    #         self.view.addItem(self.graphs[-1])
    #         self.graphs[-1].setData(pos=pos, adj=adj, size=10, symbol='o', symbolBrush=(50, 50, 200, 00), pxMode=False)



    def btn_autoLabel_clicked(self):
        print("btn_autoLabel_clicked Not Implemented")
        # raise NotImplementedError

    def btn_saveCSV_clicked(self):
        print("btn_showCSV_clicked Not Implemented")
        # raise NotImplementedError

    def btn_readCSV_clicked(self):
        print ("btn_readCSV_clicked Not Implemented")
        # raise NotImplementedError

    def btn_free3button_clicked(self):
        print ("on_free1button_clicked Not Implemented")
        # raise NotImplementedError


import cv2

class Viewer(Widget):
    def __init__(self, process):
        self.timer = QtCore.QTimer()
        self.timer.setSingleShot(True)
        self.process = process
        self.app = QtWidgets.QApplication(sys.argv)
        self.app.setApplicationName('Viewer')
        self.timer.timeout.connect(self.run_timer)
        self.timer.start(1)
        super().__init__()

    def open(self):
        self.app.exec_()

    def close(self):
        print('bye')
        sys.exit(self.app)

    def run_timer(self):

        go = self.process(self, self.qt_get_keypress)
        # go = self.process(self, self.imageWidget.qt_get_keypress)
        if go:
            self.timer.start(1)
        else:
            self.close()

    # def setCurrentImages(self, g_images, with_image):
    #     setGImages(g_images, with_image)
    #     self.setDataChanged(False)

    # def setCurrentImage(self, img=None):
    #
    #     # self.imageWidget.viewCurrentImage(img)
    #     # self.imageWidget.set_graph_points()
    #     self.viewCurrentImage(img)
    #     self.set_graph_points()

    def btn_autoLabel_clicked(self):
        self.setDataChanged(True)
        pos = mask2pos(getGImages().horizon)
        self.set_graph_points(pos, name='Horizon')


    def btn_saveCSV_clicked(self):
        print("btn_saveCSV_clicked")
        self.saveCSV()
        if self.cbx_saveTrainMask.checkState():
            self.saveTrainMask()

    def saveCSV(self):
        # print(self.graph.data['pos'])
        # pos = self.graphs[0].data['pos']
        name = self.graphs[0].name
        (cols, rows) = self.display_image.shape[:2]
        # pos[:, 0] = pos[:, 0] / cols
        # pos[:, 1] = pos[:, 1] / rows
        file_path = getGImages().file_path
        # file_name = Path(file_path).name
        # old_writecsv(file_path, pos, name)
        writecsv(file_path, self.graphs, cols, rows)
        self.setDataChanged(False)
        # self.set_graph_points(pos)


    def btn_readCSV_clicked(self):
        print("btn_readCSV_clicked")
        self.readCSV()

    def readCSV(self):
        file_path = getGImages().file_path
        _pos =  readcsv(file_path)

        if _pos is None:
            self.setDataChanged(True)
            ret = False
        else:
            for graph in self.graphs:
                self.view.removeItem(graph)
                del graph
            self.graphs = []
            # name = _pos[0][0]
            # _pos = _pos[0][1]
            # if name == 'Horizon':
            for p in _pos:
                name = p[0]
                _p = p[1]
                graph = self.new_graph(_p, name=name)
                graph.setDataChanged(False)
                # self.setDataChanged(False, name=name)
                ret = True

        return ret

    def saveTrainMask(self):
        path = Path(getGImages().file_path)
        jpgDir = path.parents[1]/pms.jpgDir
        Path.mkdir(jpgDir, exist_ok=True)
        maskDir = path.parents[1]/pms.maskDir
        Path.mkdir(maskDir, exist_ok=True)
        filename = path.name
        imgrgb = cv2.cvtColor(getGImages().small_rgb, cv2.COLOR_RGB2BGR)
        (rows, cols) = imgrgb.shape[:2]
        pts = self.graph[0].data['pos'].copy()
        pts[:, 0] = pts[:, 0] / cols
        pts[:, 1] = pts[:, 1] / rows
        pts[:,1] = 1-pts[:,1]   # invert the y axis

        (rows, cols) = (320, 480)
        output = cv2.resize(imgrgb, (cols, rows))
        fn = str((jpgDir / filename).with_suffix('.jpg'))
        cv2.imwrite(fn, output)
        scale = np.asarray(output.shape[-2:-4:-1])
        pts = (pts*scale).astype('int32')

        mask = np.zeros((rows, cols)).astype('uint8')
        mask = cv2.fillPoly(mask, pts=[pts], color=120)
        fn = str((maskDir/filename).with_suffix('.png'))
        cv2.imwrite(fn, mask)
        cv2.imshow('mask', mask)
        logger.info(f"Writing jpg and mask {fn} + jpg")


def old_writecsv(file_path, _pos):
    file_path = Path(file_path)
    filename = file_path.name
    try:
        with open(file_path.with_suffix('.txt'), 'w') as f:
            write = csv.writer(f)
            write.writerows([['Filename', filename]])
            write.writerows([['Name', 'Horizon']])
            write.writerows(_pos)
        logger.info(f"Writing CSV {file_path.with_suffix('.txt')}")
    except Exception as e:
        logger.info(e)


def writecsv(file_path, graphs, cols, rows):
    file_path = Path(file_path)
    filename = file_path.name

    try:
        with open(file_path.with_suffix('.txt'), 'w') as f:
            write = csv.writer(f)
            write.writerows([['Filename', filename]])
            for graph in graphs:
                pos = graph.data['pos'].copy()
                name = graph.name
                pos[:, 0] = pos[:, 0] / cols
                pos[:, 1] = pos[:, 1] / rows
                write.writerows([['Name', name]])
                write.writerows(pos)
        logger.info(f"Writing CSV {file_path.with_suffix('.txt')}")
    except Exception as e:
        logger.info(e)


def readcsv(file_path):
        file_path = Path(file_path)
    # try:
        with open(file_path.with_suffix('.txt'), 'r') as file:        # self.imageWidget = Widget()
            reader = csv.reader(file)
            objects = []
            lines = []
            currentName = None
            for row in reader:
                if len(row) == 0 :
                    continue
                if row[0] == 'Filename' or row[0] == 'File':
                    filename = row[1]
                    continue
                elif row[0] == 'Name':
                    if currentName is not None:
                        nplines = [[float(c) for c in r] for r in lines[1:]]
                        nplines = np.array(nplines)
                        objects.append([currentName, nplines])
                    currentName = row[1]
                    lines = []

                lines.append(row)
                # print(row)
            nplines = [[float(c) for c in r] for r in lines[1:]]
            nplines = np.array(nplines)
            objects.append([currentName, nplines])
        # if lines[0][o] == 'Filename':
        #     filename = lines[0][1]

        # lines = [[float(c) for c in r] for r in lines[1:] ]
        # lines = np.array(lines)

            return objects
    # except Exception as e:
    #     logger.warning(e)
    #     return None

def mask2pos(mask):
    n_points = pms.NUM_HORIZON_POINTS
    (rows, cols) = mask.shape
    c_pnts = np.linspace(0, cols - 1, n_points + 1)
    c_pnts[-1] = cols-1
    pos = []
    for c in c_pnts:
        # find row in column c that is non zero
        vcol = mask[::-1,int(c)]
        hpnt = np.argmax(vcol==0)
        pos.append([c/cols, hpnt/rows])

    pos.append([c / cols, 0.0])
    pos.append([0.0, 0.0])
    return np.asarray(pos)
    # return


from utils.horizon import *
if __name__ == '__main__':
    # g_images = Images()
    # image = np.random.normal(size=(320, 480, 3), loc=1024, scale=64).astype(np.uint16)

    # images = Images()
    # image = cv2.cvtColor(cv2.imread())
    # filename = '/home/jn/data/Tairua_15Jan2022/109MSDCF/DSC03288.JPG'
    filename = '/home/jn/data/testImages/original/DSC01013.JPG'
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_RGB2BGR)

    setGImages(image, filename)
    getGImages().mask_sky()
    gray_img_s = getGImages().small_gray.copy()
    getGImages().horizon = set_horizon(gray_img_s)
    cv2.imshow('horizon', getGImages().horizon)

    cv2.imshow('mask_sky', getGImages().mask)

    pos = mask2pos(getGImages().horizon)

    def test(_viewer, qt_get_keypress):
        # _viewer.update_image(images)

        # cv2.imshow('small_rgb', cv2.cvtColor(getGImages().small_rgb, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(100)

        if k == ord('q') or k == 27:
            return False

        return True

    print('here')
    viewer = Viewer(test)
    # viewer.setCurrentImage()
    viewer.readCSV()
    # viewer.viewCurrentImage(getGImages().mask)
    viewer.open()
    viewer.close()

