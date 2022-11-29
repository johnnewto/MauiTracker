__all__ = ['Viewer']

import json
from math import cos, sin, radians
from typing import List

import pyqtgraph as pg
import sys

from PyQt5.QtCore import QEvent, Qt, QCoreApplication
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
from utils.horizon_ROI import HorizonROI

import logging
logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=pms.LOGGING_LEVEL)
logger = logging.getLogger(__name__)

labels = pms.labels

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
        self.parent = parent
        # self.parent: Widget = parent
        # self.getMenu()
        self.rois = []
        self.selected_roi = None
        self._currentClassLabel = list(pms.labels.keys())[0]
        self._currentROISize = None

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            if self.raiseContextMenu(ev):
                ev.accept()

    def raiseContextMenu(self, ev):
        self.downPos = ev.pos()
        menu = self.getMenu()
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(int(pos.x()), int(pos.y())))
        return True

    def getMenu(self):
        """
        Create the menu
        """
        if self.menu is None:
            self.menu = QtWidgets.QMenu()
            self._addregion = QtGui.QAction("Add Region", self.menu)
            self._addregion.triggered.connect(self.graph_addregion)
            self.menu.addAction(self._addregion)

        return self.menu

    def graph_addregion(self):
        print("graph_addregion", self.downPos)
        pos = np.array([self.downPos.x(), self.downPos.y()])
        label = self.get_current_class_label()
        size = self.get_current_ROI_size()

        self.new_roi(pos=pos, label=label, size=size, type='rectroi', name=None)

        # # self.parent.set_graph_points(pos, name="RECT")
        # self.parent.rois.append()
        # data = self.data
        # pos = list(data['pos'])
        # data['pos'] = np.append(data['pos'], [[128,128]], axis=0)
        # data['adj'][-1] = np.array([3,4])
        # data['adj'] = np.append(data['adj'], [[4,0]], axis=0)
        # pass

    def lockX(self, roi, n, value):
        state = roi.stateCopy()
        ang = state['angle']
        ang = radians(ang)
        Rp = np.array([[cos(ang), -sin(ang)], [sin(ang), cos(ang)]])
        Rm = np.array([[cos(-ang), -sin(-ang)], [sin(-ang), cos(-ang)]])
        pos = np.array(state['pos'])
        point = np.array([roi.handles[n]['pos'].x(), roi.handles[n]['pos'].y()])
        # map back to the image and set x to zero
        imgpoint = point @ Rp.T + pos
        imgpoint[0] = value
        newpoint = (imgpoint - pos) @ Rm.T
        # print(imgpoint, newpoint, point)
        roi.handles[n]['pos'] = QtCore.QPointF(*newpoint)

    def sig_horiz_roi_update(self, roi):
        if isinstance(roi, HorizonROI):
            self.setROIChanged(roi, True)
            self.lockX(roi, 0, 0.0)
            self.lockX(roi, -3, 6000.0)
            roi.dataChanged = True
            # self.set_class_btn(roi)
            self.sig_roi_update(roi)

    def sig_roi_remove(self, roi):
        logger.info('roi_remove')
        self.parent.view.removeItem(roi)
        self.rois.remove(roi)
        self.selected_roi = None
        self.set_save_btn(False)
        # for _roi in self.rois:
        #     if _roi is roi:
        #         self.rois

    def sig_roi_update(self, roi):
        self.setAllROIChanged(False)
        self.setROIChanged(roi, True)
        self.set_class_btn(roi)
        self.selected_roi = roi
        self.set_save_btn(False)

    def sig_roi_clicked(self, roi):
        logger.info('roi_clicked ' + roi.state['label'])
        self.set_current_class_label(roi.state['label'])
        self.set_current_ROI_size(roi.state['size'])
        self.sig_roi_update(roi)
        self.set_save_btn(False)


    def sig_roi_hover(self, roi):
        pass
        # logger.info('roi_hover')

    def setAllROIChanged(self, _bool, name=None):
        for roi in self.rois:
            if name is None or roi.name == name:
                roi.dataChanged = _bool
                self.setROIChanged(roi, _bool)

    def setROIChanged(self, roi, _bool):
        label = roi.state['label']
        color = labels[label]
        if _bool:
            roi.setPen(pg.mkPen(color, width=3))
            # self.set_save_btn(False)
        else:
            roi.setPen(pg.mkPen(color, width=2))
        roi.dataChanged = _bool

    def getDataChanged(self):
        for roi in self.rois:
            if roi.dataChanged:
                    return True
        return False

    def removeAllROI(self):
        for roi in self.rois:
            self.parent.view.removeItem(roi)
        self.rois = []

    def set_class_btn(self, roi):
        # color = pms.labels[roi.state['label']] if _bool else "background-color: light gray"
        for btn in self.parent.classButtons:
            if btn.text() == roi.state['label']:
                color = pms.labels[roi.state['label']]
                btn.setStyleSheet(f"background-color: {color}")
            else:
                btn.setStyleSheet("background-color: light gray")

    def get_current_class_label(self):
        return self._currentClassLabel

    def set_current_class_label(self, label):
        self._currentClassLabel = label

    def get_current_ROI_size(self):
        return self._currentROISize

    def set_current_ROI_size(self, size):
        self._currentROISize = size

    def set_save_btn(self, _bool):
        self.parent.set_save_btn(_bool)


    def removeHorizonROI(self):
        for roi in self.rois:
            if isinstance(roi, HorizonROI):
                self.parent.view.removeItem(roi)
        self.rois = [roi for roi in self.rois if not isinstance(roi, HorizonROI)]

    def new_roi(self, points=None, pos=None, size=None, label=None, type=None, name=None, norm=False):
        # if pos is None or pos.shape[0] == 0:
        #     return
        # else:
        #     n_points = pos.shape[0]
        #
        if norm:
            (cols, rows) = self.image.shape[:2]
            if points is not None:
                points[:,0] = points[:,0] * cols
                points[:,1] = points[:,1] * rows
                points = points.astype('int32')
            if pos is not None:
                pos[0] = pos[0] * cols
                pos[1] = pos[1] * rows
                pos = pos.astype('int32')
            if size is not None:
                size[0] = size[0] * cols
                size[1] = size[1] * rows
                size = size.astype('int32')
            # assert max(pos[:,0]) < cols , 'columns must be less than image width '
        if size is None:
            size = [200,200]
        if type.lower() == 'rectroi':
            roi = pg.RectROI(pos, size, pen=(0, 9), removable=True)
            roi.sigRegionChanged.connect(self.sig_roi_update)
            roi.sigRemoveRequested.connect(self.sig_roi_remove)
            roi.sigClicked.connect(self.sig_roi_clicked)
            roi.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)

        elif type.lower() == 'horizonroi' or type.lower() == 'polylineroi':
            roi = HorizonROI(points, pos=pos, pen=pg.mkPen('g', width=2), closed=False, movable=False, removable=True)
            # roi.state['label'] = 'Horizon'

            roi.addRotateFreeHandle((points[0]+points[1])//2, points[-1])
            roi.addRotateFreeHandle((points[-1]+points[-2])//2, points[0])
            # roi.setPoints(pos)
            roi.sigRegionChanged.connect(self.sig_horiz_roi_update)
            roi.sigRemoveRequested.connect(self.sig_roi_remove)
            roi.sigClicked.connect(self.sig_roi_clicked)
            roi.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
            roi.sigHoverEvent.connect(self.sig_roi_hover)

        else:
            logger.warning(f'type {name} is not known')
        if name is not None:
            roi.name = name
        if label is None:
            roi.state['label'] = list(pms.labels.keys())[0]
        else:
            roi.state['label'] = label
        self.parent.img.setROIChanged(roi, False)
        roi.dataChanged = False
        self.parent.view.addItem(roi)
        self.rois.append(roi)
        self.sig_roi_update(roi)   # make selected etc


class Widget(QtWidgets.QWidget):
    def __init__(self):
        super(Widget, self).__init__()
        self.init_ui()
        self.qt_connections()
        self._key_press = None
        self.viewCurrentImage()

    def init_ui(self):

        self.setWindowTitle('Viewer')
        self.win = pg.GraphicsLayoutWidget()
        # self.img = pg.ImageItem(border='w')
        self.img = DrawingImage(self)
        # self.graphs = []

        # self.graphs = [Graph(name='Horizon')]
        # self.graphnames = ['obj1']
        self.view = self.win.addViewBox()
        self.view.addItem(self.img)
        self._state = self.view.getState()
        self._state['aspectLocked'] = True
        self.view.setState(self._state)
        # self.view.setLimits(xMin=-1000, xMax=7000, yMin=-1000, yMax=5000, minXRange=1000, maxXRange=10000)
        # self.view.setLimits(xMin=-1000, xMax=6000, yMin=-1000, yMax=5000, minXRange=200)
        # self.setPanLimits()
        self.btn_autoLabel = MyPushButton("Auto Horizon")
        self.btn_saveROIS = MyPushButton("Save ROIS")
        self.btn_readROIS = MyPushButton("Read ROIS")
        self.btn_free3 = MyPushButton("Free")
        self.btn_next = MyPushButton("Next")
        self.btn_back = MyPushButton("Back")
        self.btn_openDir = MyPushButton("Open Directory")
        self.cbx_saveROIs = QtWidgets.QCheckBox("Save ROIs")


        # self.cbx_autoSaveROIs = QtWidgets.QCheckBox("Auto Save ROIs")

        vertButtonBarLayout1 = QtWidgets.QVBoxLayout()
        vertButtonBarLayout1.addWidget(self.btn_autoLabel)
        vertButtonBarLayout1.addWidget(self.cbx_saveROIs)
        vertButtonBarLayout1.addWidget(self.btn_saveROIS)
        vertButtonBarLayout1.addWidget(self.btn_readROIS)
        vertButtonBarLayout1.addWidget(self.btn_next)
        vertButtonBarLayout1.addWidget(self.btn_back)
        vertButtonBarLayout1.addWidget(self.btn_free3)
        vertButtonBarLayout1.addWidget(self.btn_openDir)

        vertButtonBarLayout1.setAlignment(Qt.AlignTop)

        horzImgWinLayout = QtWidgets.QHBoxLayout()
        horzImgWinLayout.addWidget(self.win)
        horzImgWinLayout.addLayout(vertButtonBarLayout1)

        horzButtonBarLayout1 = QtWidgets.QHBoxLayout()
        self.classButtons = []
        if labels is not None:
            for lab in labels:
                btn = MyPushButton(lab)
                btn.setObjectName(lab)
                # btn.setStyleSheet(f"background-color: {labels[lab]}")
                horzButtonBarLayout1.addWidget(btn)
                self.classButtons.append(btn)

        self.set_class_btn_colors()

        verticalLayout = QtWidgets.QVBoxLayout()
        verticalLayout.addLayout(horzImgWinLayout)
        verticalLayout.addLayout(horzButtonBarLayout1)
        self.setLayout(verticalLayout)
        self.setGeometry(100, 100, 1500, 1000)

        self.text = pg.TextItem("Hello",
            # html='<div style="text-align: center"><span style="color: #FFF;">This is the</span><br><span style="color: #FF0; font-size: 16pt;">PEAK</span></div>',
            anchor=(-0.3, 0.5),  fill=(0, 0, 0, 100))
        self.view.addItem(self.text)
        self.text.setPos(1000, 4100)
        # self.text.setPointSize(16)
        self.text.setPlainText('Filename')
        self.show()

    def setPanLimits(self, xy=1000):   #jn
        (c,r,z) = self.img.image.shape
        self.view.setLimits(xMin=-1000, xMax=c+1000, yMin=-1000, yMax=r+1000, minXRange=200)

    def setImageLabel(self, text=''):
        self.text.setPlainText(text)

    def qt_connections(self):
        self.btn_autoLabel.clicked.connect(self.btn_autoLabel_clicked)
        self.btn_saveROIS.clicked.connect(self.btn_saveROIS_clicked)
        self.btn_readROIS.clicked.connect(self.btn_readROIS_clicked)
        # self.btn_next.clicked.connect(self.btn_next_clicked)
        # self.btn_back.clicked.connect(self.btn_back_clicked)
        self.btn_openDir.clicked.connect(self.btn_openDir_clicked)

        self.btn_free3.clicked.connect(self.btn_free3button_clicked)
        for btn in self. classButtons:
            btn.clicked.connect(self.btn_set_current_roi_label)

    def set_class_btn_colors(self):
        for btn in self. classButtons:
            btn.setStyleSheet(f"background-color: {labels[btn.text()]}")

    def getClassButtonActive(self):
        for btn in self. classButtons:
            btn.setStyleSheet(f"background-color: {labels[btn.text()]}")

    def set_save_btn(self, _bool):
        if _bool is None:
            self.btn_saveROIS.setStyleSheet("background-color: light gray")
            return
        if _bool:
            self.btn_saveROIS.setStyleSheet("background-color: green")
        else:
            self.btn_saveROIS.setStyleSheet("background-color: red")


    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q or event.key() == QtCore.Qt.Key_Escape :
            self.deleteLater()

        self._key_press = event.key()
        event.accept()

    def qt_get_keypress(self):
        ret = self._key_press
        self._key_press = None
        return ret

    def labelImage(self):
        filename = Path(getGImages().file_path)
        filename_rois = filename.with_suffix('.txt')
        if not filename_rois.is_file():
            filename_rois = Path('')
        self.setWindowTitle(f'{filename.name}')
        self.setImageLabel(f'{filename.name}    :     {filename_rois.name}')

    def viewCurrentImage(self, img=None):
        try:
            if img is None:
                img = getGImages().full_rgb
            out = cv2.transpose(img)
            out = cv2.flip(out, flipCode=1)
            self.img.setImage(out)
            self.setPanLimits(xy=1000)
            self.img.viewTransform()
            self.labelImage()
        except Exception as e:
            logger.error(e)
            pass

    def btn_set_current_roi_label(self):
        # find selected roi and chane label
        roi = self.img.selected_roi
        self.img.set_current_class_label(self.sender().text())
        if roi is not None:
            roi.state['label'] = self.sender().text()
            self.img.sig_roi_update(roi)


    def btn_autoLabel_clicked(self):
        print("btn_autoLabel_clicked Not Implemented")
        # raise NotImplementedError

    def btn_saveROIS_clicked(self):
        print("btn_saveROIS_clicked Not Implemented")
        # raise NotImplementedError

    def btn_readROIS_clicked(self):
        print ("btn_readROIS_clicked Not Implemented")
        # raise NotImplementedError
    def btn_next_clicked(self):
        print ("btn_next_clicked Not Implemented")
        # raise NotImplementedError
    def btn_back_clicked(self):
        print ("btn_back_clicked Not Implemented")
        # raise NotImplementedError
    def btn_openDir_clicked(self):
        print ("btn_openDir_clicked Not Implemented")
        # raise NotImplementedError


    def btn_free3button_clicked(self):
        print ("on_free1button_clicked Not Implemented")
        self.view.setState(self._state)
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
        # self.addLabel("<b>Standard mouse interaction:</b><br>left-drag to pan, right-drag to zoom.")


    def open(self, param_tree=None):
        if param_tree is not None:
            param_tree.show()
            param_tree.p.sigTreeStateChanged.connect(self.param_change)
            self.param_tree = param_tree
        self.app.exec_()

    # def close(self):
    #     print('bye')
    #     # self.param_tree.close()
    #     sys.exit(self.app)

    def close(self):
        print('Closing Viewer')
        self.win.close()
        # self.timer_stop()
        QCoreApplication.quit()

    def run_timer(self):

        go = self.process(self, self.qt_get_keypress)
        # go = self.process(self, self.imageWidget.qt_get_keypress)
        if go:
            self.timer.start(1)
        else:
            self.close()

    def getDataChanged(self):
        return self.img.getDataChanged()

    def btn_autoLabel_clicked(self):
        self.setFocus()
        self.img.setAllROIChanged(True)
        pos = mask2pos(getGImages().horizon)
        # remove existing HorizonROI
        self.img.removeHorizonROI()
        self.img.new_roi(pos, type='HorizonROI', label='Horizon', norm=False)
        self.setFocus()   # otherwise button has key focus

    def btn_saveROIS_clicked(self):
        print("btn_saveROIS_clicked")
        if self.cbx_saveROIs.checkState():
            self.saveTrainMask()
            self.saveROIS()
            self.labelImage()
            self.set_save_btn(True)
        else:
            print("Nothing Saved: cbx_saveROIs is not set")

        self.setFocus()   # otherwise button has key focus

    def saveROIS(self):
        self.set_class_btn_colors()
        (cols, rows) = self.img.image.shape[:2]
        file_path = getGImages().file_path
        write_rois(file_path, self.img.rois, cols, rows)
        self.img.setAllROIChanged(False)

    def btn_readROIS_clicked(self):
        print("btn_readROIS_clicked")
        self.read_rois()
        self.setFocus()   # otherwise button has key focus

    def btn_openDir_clicked(self):
        print("btn_openDir_clicked")
        # self.read_rois()
        filepath = Path(getGImages().file_path)
        dir_ = QtGui.QFileDialog.getExistingDirectory(None, 'Select a folder:', str(filepath.parent),
                                                      QtGui.QFileDialog.ShowDirsOnly)

        logger.info(f"Directory Selected {dir_}")
        self.setFocus()  # otherwise button has key focus

    def saveTrainMask(self):
        print(" SaveTrainMask : Not implemented ")
        return
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


    ## If anything changes in the paremeter tree, print a message
    def param_change(self, param, changes):
        print("tree changes:")


    def read_rois(self, norm=True):
        import re
        import json

        FLAGS = re.VERBOSE | re.MULTILINE | re.DOTALL
        WHITESPACE = re.compile(r'[ \t\n\r]*', FLAGS)

        def grabJSON(s):
            """Takes the largest bite of JSON from the string.
               Returns (object_parsed, remaining_string)
            """
            decoder = json.JSONDecoder()
            obj, end = decoder.raw_decode(s)
            end = WHITESPACE.match(s, end).end()
            return obj, s[end:]

        self.set_class_btn_colors()

        file_path = getGImages().file_path

        self.img.setAllROIChanged(True)
        try:
            with open(Path(file_path).with_suffix('.txt'), 'r') as file:
                self.img.removeAllROI()  # file found so delete exsiting rois
                # with open(file_path) as f:
                s = file.read()
                s = s.replace("\'", "\"")

            while True:
                obj, remaining = grabJSON(s)
                # setState
                pos = obj['pos']
                pos = np.array(pos)
                size = obj['size']
                size = np.array(size)
                ang = obj['angle']
                ang = radians(ang)
                ## create rotation transform
                type = obj['type']

                points = None
                if type == "PolyLineROI" or type == "HorizonROI":
                    points = obj['points']
                    R = np.array([[cos(ang), -sin(ang)], [sin(ang), cos(ang)]])
                    points = np.array(points)
                    points = points @ R.T + pos
                if "name" in obj:
                    name = obj['name']
                else:
                    name = None

                self.img.new_roi(points, pos=pos, size=size, label=obj['label'], type=type, name=name, norm=norm)

                s = remaining
                if not remaining.strip():
                    break
            self.img.setAllROIChanged(False)
            self.set_save_btn(True)
        except Exception as e:
            logger.warning(f"Error Reading label file {Path(file_path).with_suffix('.txt')} : {e}")
            self.set_save_btn(False)
            return None

    # def setText(self, text=''):
    #     pg.TextItem(text, color=(200, 200, 200), html=None, anchor=(0, 0), border=None, fill=None,
    #             angle=0, rotateAxis=None)

def write_rois(file_path, rois, cols, rows, norm=True):
    file_path = Path(file_path)
    filename = file_path.name

    try:
        with open(file_path.with_suffix('.txt'), 'w') as file:
            for roi in rois:
                d = roi.saveState()
                # d['filename'] = filename
                if isinstance(roi, pg.PolyLineROI):
                    d['type'] = "PolyLineROI"
                elif isinstance(roi, HorizonROI):
                    d['type'] = "HorizonROI"
                elif isinstance(roi, pg.RectROI):
                    d['type'] = "RectROI"

                d['label'] = roi.state['label']
                if norm:
                    if isinstance(roi, pg.PolyLineROI) or isinstance(roi, HorizonROI):
                        d['points'] = [(_d[0] / cols, _d[1] / rows) for _d in d['points']]
                    d['pos'] = (d['pos'][0] / cols, d['pos'][1] / rows)
                    d['size'] = (d['size'][0] / cols, d['size'][1] / rows)
                json.dump(d, file, indent=4)

        logger.info(f"Writing label file {file_path.with_suffix('.txt')}")
    except Exception as e:
        logger.warning(f"Error Writing label file {file_path.with_suffix('.txt')} : {e}")

def mask2pos(mask, addbottom=False):
    n_points = pms.NUM_HORIZON_POINTS
    (rows, cols) = mask.shape
    # (rows, cols) = (4000, 6000)
    c_pnts = np.linspace(0, cols - 1, n_points + 1)
    c_pnts[-1] = cols-1
    pos = []
    for c in c_pnts:
        # find row in column c that is non zero
        vcol = mask[::-1,int(c)]
        hpnt = np.argmax(vcol==0)
        # pos.append([c/cols, hpnt/rows])
        pos.append([c, hpnt])
    if addbottom:
        # pos.append([c / cols, 0.0])
        pos.append([c, 0.0])
        pos.append([0.0, 0.0])
    pos = np.asarray(pos) * 12
    return pos
    # return


from utils.horizon import *
if __name__ == '__main__':
    # g_images = Images()
    # image = np.random.normal(size=(320, 480, 3), loc=1024, scale=64).astype(np.uint16)

    # images = Images()
    # image = cv2.cvtColor(cv2.imread())
    # filename = '/home/jn/data/Tairua_15Jan2022/109MSDCF/DSC03288.JPG'
    filename = '/home/jn/data/test2/DSC01013.JPG'
    # filename = '/home/jn/data/test2/vlcsnap1.png'
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_RGB2BGR)

    setGImages(image, filename)
    getGImages().mask_sky()
    gray_img_s = getGImages().small_gray.copy()
    getGImages().horizon = set_horizon(gray_img_s)

    pos = mask2pos(getGImages().horizon)

    def test(_viewer, qt_get_keypress):
        k = cv2.waitKey(100)

        if k == ord('q') or k == 27:
            return False

        return True

    print('here')
    viewer = Viewer(test)
    viewer.viewCurrentImage()
    viewer.read_rois()
    viewer.open()