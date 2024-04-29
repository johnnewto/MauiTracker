"""
This file contains the main code for the MauiTracker application.
The code includes classes and functions for processing and analyzing video frames.

Classes:
- Main2: Represents the main class for processing video frames.
- Main: Represents the main class for processing and analyzing video frames.

Functions:
- run: Runs the video processing and analysis.
- experiment: Performs an experiment on the input image.

Variables:
- testpoint: A variable used for testing purposes.

Modules:
- math: Provides mathematical functions.
- time: Provides functions for working with time.
- cv2: Provides computer vision functions.
- pathlib: Provides classes for working with file paths.
- numpy: Provides functions for working with arrays.
- socket: Provides functions for working with sockets.
- csv: Provides functions for working with CSV files.
- os: Provides functions for working with the operating system.
- mot_sort_2: Contains the Sort class for object tracking.
- cmo_peak: Contains the CMO_Peak class for object detection.
- draw_tracks: Contains functions for drawing object tracks.
- g_images: Contains functions for working with images... uses globals  :(
- horizon: Contains functions for detecting the horizon in an image.
- image_utils: Contains utility functions for image processing.
- show_images: Contains functions for displaying images.
- image_loader: Contains the ImageLoader class for loading images.
- parameters: Contains various parameters for the application.
- logging: Provides functions for logging.
- sony_cam: Contains functions for working with Sony cameras.
- basler_camera: Contains functions for working with Basler cameras.
"""

import math
import time
import cv2
from pathlib import Path
import numpy as np
import socket
import csv, os


from utils.cmo_peak import *


from utils.g_images import *
from utils.horizon import *
from utils.image_utils import *
from utils.show_images import *
import utils.image_loader as il
try:
    # optional installs
    from motrackers.utils import draw_tracks
    from mot_sort_2 import Sort
    import utils.sony_cam as sony
    import utils.basler_camera as basler
except:
    pass


from utils import parameters as pms
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

    def run(self, wait_timeout=1):

        for j, (idx, image) in enumerate(iter(loader)):
            print("frame", idx)
            if j == 10:
                break


class Main:
    def __init__(self, _loader, model, tracker, display_width=2000, record=False, path='', qgc=None):
        self.loader = _loader
        self.model: CMO_Peak = model
        self.tracker = tracker
        self.display_width = display_width
        self.record = record
        self.do_run = True
        self.path = path
        self.testpoint = None
        self.all_tiles = []
        self.label_list = []

        self.model.set_max_pool(12)
        self.heading_angle = 0.0
        self.qgc: ConnectQGC = qgc
        self.sock = socket.socket(socket.AF_INET,  # Internet
                                  socket.SOCK_DGRAM)  # UDP

    def experiment(self, image):
        gray_img_s = getGImages().small_gray.copy()
        getGImages().horizon = set_horizon(gray_img_s)
        cv2_img_show('horizon', getGImages().horizon)

        # try:
        #     # imgrgb_s = resize(image, width=6000 // self.model.maxpool)
        #     # gray_img_s = cv2.cvtColor(imgrgb_s, cv2.COLOR_BGR2GRAY)
        #
        #     imgrgb_s = getGImages().small_rgb.copy()
        #
        #
        #     edges = cv2.Canny(gray_img_s, threshold1=50, threshold2=150, apertureSize=3)
        #     cv2_img_show('Canny1`', edges)
        #     kernel = np.ones((3, 5), np.uint8)
        #     edges = cv2.dilate(edges, kernel, iterations=1)  # < --- Added a dilate, check link I provided
        #     kernel = np.ones((1, 5), np.uint8)
        #     edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=5)
        #     # edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)
        #     # show_img(edges, figsize=(12, 8))
        #     # show_img(imgrgb_s, mode='BGR', figsize=(12,8))
        #     mask = np.ones_like(edges, dtype='uint8')
        #     mask[edges == 255] = 0
        #     num_regions, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        #     area_sort = np.argsort(stats[:, -1])
        #
        #     # choose the region that is the largest brightest
        #     brightest = 0
        #     sky_idx = -1
        #
        #     # get bright large area
        #     for i in range(min(num_regions, 3)):
        #         idx = area_sort[-(i + 1)]
        #         b = np.mean(gray_img_s[labels == idx])
        #         area_ratio = stats[idx][4] / (mask.shape[0] * mask.shape[1])
        #         if b > brightest and area_ratio > 0.25:
        #             brightest = b
        #             sky_idx = idx
        #
        #     assert sky_idx > -1
        #     labels[labels != sky_idx] = 0
        #     labels[labels == sky_idx] = 128
        #     labels = labels.astype('uint8')
        #     # kernel = np.ones((5, 5), 'uint8')
        #     # labels = cv2.morphologyEx(labels, cv2.MORPH_CLOSE, kernel, iterations=5)
        #     cv2_img_show('labels', labels)
        #     getGImages().horizon = labels
        #
        #     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, lines=None, minLineLength=100, maxLineGap=2)
        #     lines = list(np.squeeze(lines))
        #     # lines.sort(key=lambda x: -(x[2] - x[0]))
        #
        #     for line in lines:
        #         [x1, y1, x2, y2] = line
        #         angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        #         if angle < 25:
        #             cv2.line(imgrgb_s, (x1, y1), (x2, y2), (0, 255, 0), 1)
        #         # cv2.line(imgrgb_s, (x1, y1), (x2, y2), (0, 255, 0), 1)
        #
        #     cv2_img_show('Canny2', edges)
        #     cv2_img_show('HoughLinesP', imgrgb_s)
        # except Exception as e:
        #     logger.warning(e)

    def run(self, wait_timeout=10, heading_angle=0, stop_frame=None):
        """
            Run the main tracking loop.

            Args:
                wait_timeout (int, optional): The wait timeout in milliseconds. Defaults to 10.
                heading_angle (int, optional): The heading angle. Defaults to 0.
                stop_frame (int, optional): The frame number to stop at. Defaults to None.
            """
        self.heading_angle = heading_angle
        WindowName = "Main View"
        cv2.namedWindow(WindowName, cv2.WINDOW_NORMAL)

        # These two lines will force your "Main View" window to be on top with focus.
        cv2.setWindowProperty(WindowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(WindowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        if self.record:
            video = VideoWriter(self.path + '.mp4', fps=5.0)
            print(f"Recording to {self.path}.mp4")
        self.sock.sendto(b"Start Record", ("127.0.0.1", 5005))

        first_run = True
        while self.do_run:
            k = -1
            for (image, filename), frameNum, grabbed in iter(self.loader):

                if grabbed or first_run:
                    first_run = False
                    print(f"frame {frameNum} : {filename}  {grabbed}")
                    if len(image.shape) == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
                    setGImages(image)
                    getGImages().mask_sky()
                    getGImages().small_objects()

                    # self.experiment(image)

                    # scale between source image and display image
                    self.display_scale = self.display_width / image.shape[1]

                    # sr, sc = self.model.align()
                    self.model.detect()
                    if STORE_LABELS:
                        for i, bbox in enumerate(self.model.bbwhs):
                            self.label_list.append([filename, i, bbox[0] + bbox[2] // 2, bbox[1] + bbox[2] // 2])

                    disp_image = self.display_results(image)
                    # disp_image = self.display_results(image, frameNum, bboxes, bbwhs, confidences, class_ids, (sr, sc))
 
                    putText(disp_image, f'Frame# = {frameNum}, {filename}', row=170, fontScale=0.5)

                    if self.record:
                        # img = cv2.cvtColor(disp_image, code=cv2.COLOR_RGB2BGR)
                        img = resize(disp_image, width=3000)  # not sure why this is needed to stop black screen video
                        # img = (np.random.rand(200, 200, 3) * 255).astype('uint8')
                        for i in range(2):
                            video.add(img)

                    self.sock.sendto(b"Take Screen Shot", ("127.0.0.1", 5005))

                try:
                    cv2_img_show('fullres_tiles', vstack(
                        [np.hstack(self.model.fullres_img_tile_lst), np.hstack(self.model.fullres_cmo_tile_lst)]),
                                 height=200)

                    cv2_img_show('lowres_tiles', vstack(
                        [np.hstack(self.model.lowres_img_tile_lst), np.hstack(self.model.lowres_cmo_tile_lst)]),
                                 height=200)
                except Exception as e:
                    logger.error(e)

                cv2_img_show(WindowName, disp_image)
                # k = cv2.waitKey(wait_timeout)
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
                    self.loader._direction_fwd = not self.loader._direction_fwd
                if k == ord('r'):
                    # change direction
                    wait_timeout = 0
                    self.loader.restep = True
                # if k == ord('t'):
                #     r = int(self.images.small_rgb.shape[0] * 0.9)
                #     c =  self.model.small_rgb.shape[1] // 2
                #     bboxes.append([3000, 2000, 40, 40])       
                #     confidences.append(1)
                #     class_ids.append(2)
                #     self.tracker.update(bboxes, confidences, class_ids, (0, 0))
                #     # dets = np.array([convert_x_to_bbox([c, r, 900, 1], 0.9).squeeze() for (r, c) in pks])
                #     # trackers = mot_tracker.update(dets)
                #     self.testpoint = (r, c)
                #     self.loader.direction_fwd = not self.loader.direction_fwd

                if k == ord('f'):
                    import tkinter.filedialog

                    path = tkinter.filedialog.askdirectory()
                    self.loader.open_path(path)

                    # create_filechooser(default_path=str(Path.home()) + "/data/Karioitahi_09Feb2022/")
                if stop_frame is not None and stop_frame == frameNum:
                    logger.info(f" Early stop  at {frameNum}")
                    break
                k = cv2.waitKey(wait_timeout)

            if stop_frame is not None and stop_frame == frameNum:
                logger.info(f" Early stop  at {frameNum}")
                break
            # self.loader.direction_fwd = not self.loader.direction_fwd
            wait_timeout = 0

            k = cv2.waitKey(wait_timeout)
            if k == ord('q') or k == 27:
                break

        if self.record:
            video.close()
        self.sock.sendto(b"End Record", ("127.0.0.1", 5005))

        cv2.destroyAllWindows()
        self.loader.close()

        time.sleep(0.5)

    def overlay_mask(self, image, mask, color=(255,255,0), alpha=0.4):
        """
        Overlay the mask on the image.

        Args:
            image (numpy.ndarray): The image.
            mask (numpy.ndarray): The mask.

        Returns:
            numpy.ndarray: The image with the mask overlayed.
        """
        # convert to rgb color
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask_rgb = np.where(mask_rgb == [255, 255, 255], np.uint8(color), mask_rgb)

        # resize mask to image size
        mask_rgb = cv2.resize(mask_rgb, (image.shape[1], image.shape[0]))
        return cv2.addWeighted(mask_rgb, alpha, image, 1 , 0)
    

    def draw_bboxes(self, image, bboxes, confidences, text=True, thickness=6, alpha:typ.Union[float, None]=0.3):
        """
        Draw the bounding boxes about detected objects in the image. Assums the bb are sorted by confidence

        Args:
            image (numpy.ndarray): Image or video frame.
            bboxes (numpy.ndarray): Bounding boxes pixel coordinates as (xmin, ymin, width, height)

        Returns:
            numpy.ndarray: image with the bounding boxes drawn on it.
        """

        # support for alpha blending overlay
        overlay = image.copy() if alpha is not None else image    
        count = 0
        for bb in bboxes:
            clr = (255, 0, 0) if count < 5 else (0,255,0) if count < 10 else (0, 0, 255) # in order red, green, blue
            cv2.rectangle(overlay, (bb[0], bb[1] ), (bb[0] + bb[2], bb[1] + bb[3]), clr, thickness)
            if False:
                _font_size = 1.0
                _thickness = 2
                label = f"{count}"
                (label_width, label_height), baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, _font_size, _thickness)
                y_label = max(bb[1], label_height)
                cv2.rectangle(overlay, (bb[0], y_label - label_height), (bb[0] + label_width, y_label + baseLine),
                             (0, 0, 0), cv2.FILLED)
                cv2.putText(overlay, label, (bb[0], y_label+5), cv2.FONT_HERSHEY_SIMPLEX, _font_size, clr, _thickness)
            if text:
                self.putText(overlay, f'{count}', (bb[0], bb[1]-0), fontScale=1.0, color=clr, thickness=2)
            count += 1
        
        if alpha is not None:
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        return image

    # def puttext(self, image, text, row, col, fontScale=0.5, color=(255, 255, 255), thickness=1):
    #     """
    #     Put text on the image."""

    def putText(self, img, text, position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), thickness=2):
        """
        Put text on the image with a black background.

        Args:
            img (numpy.ndarray): The image.
            text (str): The text to put on the image.
            position (tuple): The position where the text should be put.
            fontFace (int): The font type. Default is cv2.FONT_HERSHEY_SIMPLEX.
            fontScale (float): Font scale. Default is 1.
            color (tuple): Font color. Default is white.
            thickness (int): Thickness of the lines used to draw a text. Default is 2.

        Returns:
            numpy.ndarray: The image with the text.
        """
        # Calculate the width and height of the text
        (text_width, text_height), baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)
        # Determine the y-coordinate for the text
        y_text = max(position[1], text_height)
        # Draw a filled rectangle on the image at the location where the text will be placed
        cv2.rectangle(img, (position[0], y_text - text_height), (position[0] + text_width, y_text + baseLine), (0, 0, 0), cv2.FILLED)
        # Draw the text on the image at the specified location
        cv2.putText(img, text, (position[0], y_text+text_height//4), fontFace, fontScale, color, thickness, cv2.LINE_AA)

        return img

    def display_results(self, image, alpha=0.3):

        disp_image = self.overlay_mask(image, getGImages().mask, alpha=alpha)
        disp_image = self.draw_bboxes(disp_image, self.model.bbwhs, self.model.pks, text=True, thickness=8, alpha=alpha)
        for count, tile in enumerate (self.model.fullres_img_tile_lst):
            # puttext label count in left top corner
            clr = (255, 0, 0) if count < 5 else (0,255,0) if count < 10 else (0, 0, 255)  # in order red, green, blue
            # cv2.putText(tile, str(count), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1)
            self.putText(tile, f'{count}', (0,10), fontScale=0.4, color=clr, thickness=1)
        try:
            tile_img = np.hstack(self.model.fullres_img_tile_lst)
            tile_img = resize(tile_img, width=self.display_width, inter=cv2.INTER_NEAREST)
            disp_image = np.vstack([tile_img, disp_image])
        except Exception as e:
            logger.error(e)
                                                       
        return disp_image
           
    def old_display_results(self, image, frameNum, bboxes, bbwhs, confidences, class_ids, pos):


        # the source image is very large so we reduce it to a more manageable size for display
        # disp_image = cv2.cvtColor(resize(image, width=self.display_width), cv2.COLOR_GRAY2BGR)
        # disp_image = image.copy()
        disp_image = self.overlay_mask(image, getGImages().mask, alpha=0.3)

        display_scale = (image.shape[1] / getGImages().small_gray.shape[1], image.shape[0] / getGImages().small_gray.shape[0])  

        CONTOURS = False
        if CONTOURS:
            contours, hierarchy = cv2.findContours(getGImages().mask * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # disp_image = resize(image, width=getGImages().small_gray.shape[1])
    
            # multiply the contours by the display_scale
            contours = [np.squeeze((c * display_scale).astype('int32')) for c in contours]

            for idx in range(len(contours)):
                cv2.drawContours(disp_image, contours, idx, (255, 0, 0), 5)


        # disp_image = resize(disp_image, width=self.display_width)

      
        SEND_ADSB = False
        if SEND_ADSB:
            angle_inc = sc * 54.4 / image.shape[1]
            self.heading_angle -= angle_inc
            if self.qgc is not None:
                self.qgc.high_latency2_send(heading=self.heading_angle)
                putText(disp_image, f'{self.heading_angle :.3f}', row=disp_image.shape[0] - 20, col=disp_image.shape[1] // 2)

            detections = [bb for bb, d in zip(bbwhs, class_ids) if d == 3]

            for bb in detections:
                wid = bb[2]
                rad_pix = math.radians(54.4 / image.shape[1])
                dist = 40 / (wid * rad_pix)
                heading = (bb[0] - image.shape[1] // 2) * 54.4 / image.shape[1]
                ang = self.heading_angle + heading
                print(f"plane detected at range {dist} {ang} ")
                if self.qgc is not None:
                    self.qgc.adsb_vehicle_send('bob', dist, ang, max_step=100)

            if self.testpoint is None:
                self.testpoint = (int(disp_image.shape[0] * 0.9), disp_image.shape[1] // 2)

            self.testpoint = (self.testpoint[0] + int(sr * self.display_scale), self.testpoint[1] + int(sc * self.display_scale))
            cv2.circle(disp_image, (self.testpoint[1], self.testpoint[0]), 30, (0, 255, 0), 1)

        cv2.imshow('blockreduce_mask', resize(getGImages().mask * 255, width=1000))
        cv2.imshow('blockreduce_CMO', cv2.applyColorMap(resize(norm_uint8(getGImages().cmo), width=1000), cv2.COLORMAP_MAGMA))

        TRACKER = False
        if TRACKER:
            # Tuple of 10 elements representing (frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)
            # remove the ground class
            _bboxes, _confidences, _class_ids = [], [], []
    
            for i, (bb, conf) in enumerate(zip(bboxes, confidences)):
                # if conf > 0.05:
                xyxy_conf = [bb[0] - 80, bb[1] - 40, bb[0] + 80, bb[1] + 40, conf]
                _bboxes.append(xyxy_conf)
    
            _bboxes = np.array(_bboxes)

            (sr, sc) = pos
            tracks = self.tracker.update(_bboxes, confidences, (sc, sr))

            draw_tracks(disp_image, tracks, dotsize=1, colors=self.model.bbox_colors, display_scale=self.display_scale)
 
            tiles = old_make_tile_list(image, tracks)

            tile_img = tile_images(tiles, label=True, colors=self.model.bbox_colors)
            tile_img = resize(tile_img, width=self.display_width, inter=cv2.INTER_NEAREST)
            disp_image = np.vstack([tile_img, disp_image])

        else:
            self.model.draw_bboxes(disp_image, self.model.bbwhs, None, None, display_scale=self.display_scale, text=True, thickness=8)
            tile_img = np.hstack(self.model.fullres_img_tile_lst)
            tile_img = resize(tile_img, width=self.display_width, inter=cv2.INTER_NEAREST)
            disp_image = np.vstack([tile_img, disp_image])

        if STORE_TILES:
            # logger.warning( " all_tiles.append(tiles)  will hog memory")
            for tl in tiles:
                self.all_tiles.append(tl[0])  # warning this will hog memory

        # tile_img = cv2.cvtColor(resize(tile_img, width=self.display_width), cv2.COLOR_GRAY2BGR)
        
       
        return disp_image


if __name__ == '__main__':
    import argparse
    """

    """
    STORE_TILES = False
    STORE_LABELS = False

    # RECORD = False
    STOP_FRAME = None

    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.2
    DRAW_BOUNDING_BOXES = True
    USE_GPU = False

    USE_QGC = False
    USE_TRACKER = False
    if USE_QGC:  # for position on map  ( special compiled version) should use mavlink message instead)
        from utils.qgcs_connect import ConnectQGC
    # method = 'CentroidKF_Tracker'

    # _tracker = CentroidTracker(max_lost=0, tracker_output_format='visdrone_challenge')
    # _tracker = CentroidTracker(maxDisappeared=5, maxDistance=100)

    # tracker = CentroidKF_Tracker(max_lost=0, tracker_output_format='mot_challenge')
    # tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)
    # tracker = IOUTracker(max_lost=2, iou_threshold=0.5, min_detection_confidence=0.4, max_detection_confidence=0.7,
    #                      tracker_output_format='mot_challenge')
    
    if USE_TRACKER:
        _tracker = Sort(max_age=5,
                        min_hits=3,
                        iou_threshold=0.2)
    else:
        _tracker = None

    parser = argparse.ArgumentParser(description="Your program description")
    parser.add_argument('-r', '--record', action='store_true', help='Enable recording', default=False)
    args = parser.parse_args()

    RECORD = args.record

    # home = str(Path.home())
    _model = CMO_Peak(confidence_threshold=0.1,
                      labels_path='data/imagenet_class_index.json',
                      # labels_path='/media/jn/0c013c4e-2b1c-491e-8fd8-459de5a36fd8/home/jn/data/imagenet_class_index.json',
                      expected_peak_max=60,
                      peak_min_distance=5,
                      num_peaks=10,
                      maxpool=12,
                      CMO_kernalsize=3,
                      track_boxsize=(80, 160),
                      bboxsize=40,
                      draw_bboxes=True,
                      device=None, )

    # gen_cmo = GenCMO(shape=(600,1000), dt=0.1, n=5)
    (rows, cols) = (2000, 3000)
    center = (2750, 4350)
    (_r, _c) = (center[0] - rows // 2, center[1] - cols // 2)
    crop = [_r, _r + rows, _c, _c + cols]
    # crop = None
    home = str(Path.home())

    # if data path exists use it
    path = home + '/data/maui-data/Karioitahi_09Feb2022/132MSDCF-28mm-f4' 
    # path = home + '/data/maui-data/Karioitahi_09Feb2022/136MSDCF'
    # path = home + '/data/maui-data/Karioitahi_15Jan2022/125MSDCF-landing'
    # path = home + '/data/maui-data/Tairua_15Jan2022/116MSDCF'
    # path = home + '/data/maui-data/Tairua_15Jan2022/109MSDCF'
    # path = home + '/data/maui-data/karioitahi_13Aug2022/SonyA7C/103MSDCF'
    path += '-use-local-path'
    if not os.path.exists(path):
        print(f"Path {path} does not exist, using local path")
        path = "data/Karioitahi_09Feb2022/132MSDCF-28mm-f4"

    # USE_CAMERA = 'CAM=SONY'
    # USE_CAMERA = 'CAM=BASLER'
    USE_CAMERA = 'FILES'

    # path = easygui.diropenbox( default=home+"/data/")

    if USE_CAMERA == 'CAM=SONY':
        import utils.sony_cam as sony

        try:
            cam_serials = ['00001']
            cam_names = ['FrontCentre']
            cameras = sony.RunCameras(cam_names, cam_serials, dostart=True)
            # test get_camera
            loader = cameras.get_camera('FrontCentre')
            assert loader.name == 'FrontCentre'

        except Exception as e:
            logger.error(e)
            loader = il.ImageLoader(path + '/*.JPG', mode='RGB', cvtgray=False, start_frame=0)

    elif USE_CAMERA == 'CAM=BASLER':
        import utils.basler_camera as basler

        try:
            cam_serials = ['23479535']
            cam_names = ['FrontCentre']

            cameras = basler.RunCameras(cam_names, cam_serials, dostart=True)
            # test get_camera
            loader = cameras.get_camera('FrontCentre')
            assert loader.name == 'FrontCentre'

        except Exception as e:
            logger.error(e)
            loader = il.ImageLoader(path + '/*.JPG', mode='RGB', cvtgray=False, start_frame=0)

    else:
        loader = il.ImageLoader(path + '/*.JPG', mode='RGB', cvtgray=False, start_frame=0)
    if USE_QGC:
        NZ_START_POS = (-36.9957915731748, 174.91686500754628)
        qgc = ConnectQGC(NZ_START_POS)
    else:
        qgc = None

    main = Main(loader, _model, _tracker, display_width=6000, record=RECORD, path=path, qgc=qgc)

    main.run(wait_timeout=0, heading_angle=-70, stop_frame=STOP_FRAME)

    if STORE_TILES:
        np_tiles = np.asarray(main.all_tiles)
        np.save(path + 'np_tiles.npy', np_tiles)

    if STORE_LABELS:
        def save_csv(csv_file_path):
            with open(csv_file_path, "w", newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                # write header
                writer.writerow(['img', 'Bird', 'Cloud', 'Ground', 'Plane'])
                # write one-hot labels
                for label in main.label_list:
                    label[0] = os.path.split(label[0])[-1]
                    writer.writerow(label)

            message = f'csv saved to: {csv_file_path}'
            print(message)


        path_to_save = path
        csv_file_path = path_to_save + 'assigned_classes.csv'
        save_csv(csv_file_path)

    # print(f'FPS = {loader.get_FPS()}')
