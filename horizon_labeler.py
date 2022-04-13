import math
import time
from cv2 import cv2
from pathlib import Path
import numpy as np
import socket
import csv, os

from mot_sort_2 import Sort
from motrackers import CMO_Peak, Images
from utils.qt_viewer import *
# from motrackers import CentroidTracker
# from motrackers.tracker_2 import CentroidTracker
from motrackers.utils import draw_tracks

from utils.image_utils import *
from utils.show_images import *
import utils.image_loader as il
import utils.sony_cam as sony
from utils.qgcs_connect import ConnectQGC

import easygui

from motrackers import parameters as pms
import logging
logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=pms.LOGGING_LEVEL)
logger = logging.getLogger(__name__)

testpoint = None

class Main2:
    def __init__(self, _loader, path=''):
        self.loader: il.ImageLoader = _loader
        self.path = path
        self.do_run = True

    def run(self, wait_timeout=1):
        first_run = True
        while self.do_run:
            for (image, filename), frameNum, grabbed  in iter(self.loader):
                if grabbed or first_run:
                    first_run = False
                    print(f"frame {frameNum} : {filename}  {grabbed}" )
                    # self.images = self.model.set_image(image)


        cv2.destroyAllWindows()
        self.loader.close()


class Main:
    def __init__(self, _loader, display_width=1200, record=False, path=''):
        self.loader = _loader

        self.display_width = display_width
        self.record = record
        self.do_run = True
        self.path = path
        self.testpoint = None
        self.all_tiles = []
        self.label_list = []

        self.images = Images()

    def experiment(self, image):
        try:
            # imgrgb_s = resize(image, width=6000 // self.model.maxpool)
            # gray_img_s = cv2.cvtColor(imgrgb_s, cv2.COLOR_BGR2GRAY)

            imgrgb_s = self.model.image.small_rgb.copy()
            gray_img_s = self.model.image.small_gray.copy()

            edges = cv2.Canny(gray_img_s, threshold1=50, threshold2=150, apertureSize=3)
            cv2_img_show('Canny1`', edges)
            kernel = np.ones((3, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)  # < --- Added a dilate, check link I provided
            kernel = np.ones((1, 5), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=5)
            # edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)
            # show_img(edges, figsize=(12, 8))
            # show_img(imgrgb_s, mode='BGR', figsize=(12,8))
            mask = np.ones_like(edges, dtype='uint8')
            mask[edges == 255] = 0
            num_regions, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            area_sort = np.argsort(stats[:, -1])

            # choose the region that is the largest brightest
            brightest = 0
            sky_idx = -1

            # get bright large area
            for i in range(min(num_regions, 3)):
                idx = area_sort[-(i + 1)]
                b = np.mean(gray_img_s[labels == idx])
                area_ratio = stats[idx][4] / (mask.shape[0] * mask.shape[1])
                if b > brightest and area_ratio > 0.25:
                    brightest = b
                    sky_idx = idx

            assert sky_idx > -1
            labels[labels != sky_idx] = 0
            labels[labels == sky_idx] = 128
            labels = labels.astype('uint8')
            # kernel = np.ones((5, 5), 'uint8')
            # labels = cv2.morphologyEx(labels, cv2.MORPH_CLOSE, kernel, iterations=5)
            cv2_img_show('labels', labels)
            self.model.image.horizon = labels

            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, lines=None, minLineLength=100, maxLineGap=2)
            lines = list(np.squeeze(lines))
            # lines.sort(key=lambda x: -(x[2] - x[0]))

            for line in lines:
                [x1, y1, x2, y2] = line
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                if angle < 25:
                    cv2.line(imgrgb_s, (x1, y1), (x2, y2), (0, 255, 0), 1)
                # cv2.line(imgrgb_s, (x1, y1), (x2, y2), (0, 255, 0), 1)

            cv2_img_show('Canny2', edges)
            cv2_img_show('HoughLinesP', imgrgb_s)
        except Exception as e:
            logger.warning(e)


    def get_horizon_tiles(self):
        horizon = cv2.cvtColor(self.images.horizon, cv2.COLOR_GRAY2BGR)
        small_rgb = self.images.small_rgb
        # draw vertical lines
        for c in range(20, horizon.shape[1], 20):
            horizon[:,c,2] = 255
            small_rgb[:,c,2] = 255
        for r in range(20, horizon.shape[0], 20):
            horizon[r,:,2] = 255
            small_rgb[r,:,2] = 255
        cv2_img_show('horizon', horizon)
        cv2_img_show('small_rgb', small_rgb)

    def run(self, wait_timeout=10, heading_angle=0, stop_frame=None):
        self.heading_angle = heading_angle
        self.WindowName = "Main View"
        cv2.namedWindow(self.WindowName, cv2.WINDOW_NORMAL)

        # These two lines will force your "Main View" window to be on top with focus.
        cv2.setWindowProperty(self.WindowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(self.WindowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        if self.record:
            video = VideoWriter(self.path + 'out.mp4', 15.0)


        while self.do_run:
            frameNum = self.loop(wait_timeout, stop_frame)

            if  stop_frame is not None and stop_frame == frameNum:
                logger.info(f" Early stop  at {frameNum}")
                break
            self.loader.direction_fwd = not self.loader.direction_fwd
            wait_timeout = 0

        cv2.destroyAllWindows()
        self.loader.close()
        time.sleep(0.5)

    def loop(self, wait_timeout, stop_frame):
        first_run = True
        disp_image = None
        (image, filename), frameNum, grabbed  = next(iter(self.loader))
        for (image, filename), frameNum, grabbed in iter(self.loader):
            if grabbed or first_run:
                first_run = False
                print(f"frame {frameNum} : {filename}  {grabbed}")
                self.images.set(image)
                self.experiment(image)
                # self.get_horizon_tiles()
                # scale between source image and display image
                self.display_scale = self.display_width / image.shape[1]
                self.images.mask_sky()
                disp_image = self.display_results(image, frameNum)
                putText(disp_image, f'Frame# = {frameNum}, {filename}', row=170, fontScale=0.5)

            cv2_img_show(self.WindowName, disp_image)
            k = cv2.waitKey(wait_timeout)
            if k == ord('q') or k == 27:
                self.do_run = False
                break
            if k == ord(' '):
                wait_timeout = 0
            if k == ord('g'):
                wait_timeout = 1
            if k == ord('d'):
                # change direction
                wait_timeout = 0
                self.loader.direction_fwd = not self.loader.direction_fwd
            if k == ord('r'):
                # change direction
                wait_timeout = 0
                self.loader.restep = True

            if k == ord('f'):
                import tkinter.filedialog

                path = tkinter.filedialog.askdirectory()
                self.loader.open_path(path)

                # create_filechooser(default_path=str(Path.home()) + "/data/Karioitahi_09Feb2022/")
            if stop_frame is not None and stop_frame == frameNum:
                logger.info(f" Early stop  at {frameNum}")
                break

    def display_results(self, image, frameNum):
        contours, hierarchy = cv2.findContours(self.images.mask * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        disp_image = resize(image, width=self.images.small_gray.shape[1])

        for idx in range(len(contours)):
            cv2.drawContours(disp_image, contours, idx, (255,0,0), 1)
        disp_image = resize(disp_image, width=self.display_width)
        putText(disp_image, f'{self.heading_angle :.3f}', row=disp_image.shape[0]-20, col=disp_image.shape[1]//2)
        cv2.imshow('blockreduce_mask', resize(self.images.mask * 255, width=1000))
        cv2.imshow('blockreduce_CMO', cv2.applyColorMap(resize(norm_uint8(self.images.cmo ), width=1000), cv2.COLORMAP_MAGMA))
        return disp_image

    def loop(self, wait_timeout, stop_frame):
        first_run = True
        disp_image = None
        (image, filename), frameNum, grabbed  = next(iter(self.loader))
        if grabbed:
            print(f"frame {frameNum} : {filename}  {grabbed}")
            self.images.set(image)
            self.experiment(image)
            # self.get_horizon_tiles()
            # scale between source image and display image
            self.display_scale = self.display_width / image.shape[1]
            self.images.mask_sky()
            disp_image = self.display_results(image, frameNum)
            putText(disp_image, f'Frame# = {frameNum}, {filename}', row=170, fontScale=0.5)

        cv2_img_show(self.WindowName, disp_image)
        k = cv2.waitKey(wait_timeout)
        if k == ord('q') or k == 27:
            self.do_run = False
            # break
        if k == ord(' '):
            wait_timeout = 0
        if k == ord('g'):
            wait_timeout = 1
        if k == ord('d'):
            # change direction
            wait_timeout = 0
            self.loader.direction_fwd = not self.loader.direction_fwd
        if k == ord('r'):
            # change direction
            wait_timeout = 0
            self.loader.restep = True

        if k == ord('f'):
            import tkinter.filedialog

            path = tkinter.filedialog.askdirectory()
            self.loader.open_path(path)

            # create_filechooser(default_path=str(Path.home()) + "/data/Karioitahi_09Feb2022/")
        if stop_frame is not None and stop_frame == frameNum:
            logger.info(f" Early stop  at {frameNum}")
            # break


if __name__ == '__main__':

    STORE_TILES = False
    STORE_LABELS = False

    RECORD = False
    STOP_FRAME = None

    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.2
    DRAW_BOUNDING_BOXES = True

    # gen_cmo = GenCMO(shape=(600,1000), dt=0.1, n=5)
    (rows, cols) = (2000, 3000)
    center = (2750, 4350)
    (_r, _c) = (center[0] - rows // 2, center[1] - cols // 2)
    crop = [_r, _r + rows, _c, _c + cols]
    # crop = None
    home = str(Path.home())
    # path = 'Z:/Day2/seq1/'
    path = home+"/data/large_plane/images"
    # path = home+"/data/ardmore_30Nov21"
    path = home+"/data/Karioitahi_15Jan2022/123MSDCF-35mm"
    # path = home+"/data/Karioitahi_15Jan2022/117MSDCF-28mm(subset)"
    # path = home+"/data/Karioitahi_15Jan2022/117MSDCF-28mm"
    path = home+"/data/Karioitahi_09Feb2022/132MSDCF-28mm-f4"
    # path = home+"/data/Tairua_15Jan2022/109MSDCF"

    # path = easygui.diropenbox( default=home+"/data/")

    loader = il.ImageLoader(path + '/*.JPG', mode='RGB', cvtgray=False, start_frame=0)

    # main = Main(loader, _model, _tracker, display_width=1500, record=RECORD, path=path)
    # main = Main(loader,  )
    #
    # main.run(wait_timeout=0)
    images = Images()

    def test(viewer):
        try:
            test.wait_timeout += 0
        except AttributeError:
            test.wait_timeout = 1

        k = cv2.waitKey(test.wait_timeout)

        if test.wait_timeout != 99 or k == ord(' '):
            (image, filename), frameNum, grabbed  = next(iter(loader))
            if grabbed:
                print(f"frame {frameNum} : {filename}  {grabbed}")
                images.set(image)
                image_rgb = cv2.cvtColor(images.small_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow('test', images.small_rgb)
                viewer.update_image(image_rgb)

        if k == ord('g'):
            print(k)
        if k == ord(' '):
            test.wait_timeout = 99
        # if k == ord('u'):
        #     viewer.update_image(image_rgb)
        if k == ord('q') or k == 27:
            cv2.destroyAllWindows()
            loader.close()
            return False


        return True

    viewer = Viewer(test)
    cv2.destroyAllWindows()
    loader.close()
    print(f'FPS = {loader.get_FPS()}')

    viewer.close()