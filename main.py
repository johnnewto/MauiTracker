import math
import time
import cv2
from pathlib import Path
import numpy as np
import socket
import csv, os

from mot_sort_2 import Sort
from utils.cmo_peak import *
from motrackers.utils import draw_tracks

from utils.g_images import *
from utils.horizon import *
from utils.image_utils import *
from utils.show_images import *
import utils.image_loader as il
# import utils.sony_cam as sony
# import utils.basler_camera as basler


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

        for j, (idx, image)  in enumerate(iter(loader)):
            print("frame", idx)
            if j == 10:
                break


class Main:
    def __init__(self, _loader, model, tracker, display_width=1200, record=False, path='', qgc=None):
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
        self.heading_angle = heading_angle
        WindowName = "Main View"
        cv2.namedWindow(WindowName, cv2.WINDOW_NORMAL)

        # These two lines will force your "Main View" window to be on top with focus.
        cv2.setWindowProperty(WindowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(WindowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        if self.record:
            video = VideoWriter(self.path + 'out.mp4', 15.0)
        self.sock.sendto(b"Start Record", ("127.0.0.1", 5005))

        first_run = True
        while self.do_run:
            k = -1
            for (image, filename), frameNum, grabbed  in iter(self.loader):

                if grabbed or first_run:
                    first_run = False
                    print(f"frame {frameNum} : {filename}  {grabbed}" )
                    if len(image.shape) == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
                    setGImages(image)
                    getGImages().mask_sky()

                    # self.experiment(image)

                    # scale between source image and display image
                    self.display_scale = self.display_width / image.shape[1]

                    # self.model.mask_sky()

                    bboxes, bbwhs, confidences, class_ids, (sr, sc) = self.model.detect(scale=self.display_scale,
                                                                                        filterClassID=[1, 2], frameNum=frameNum)
                    if STORE_LABELS:
                        for i, bbox in enumerate(bboxes):
                            self.label_list.append([filename, i, bbox[0]+bbox[2]//2, bbox[1]+bbox[2]//2])

                    disp_image = self.display_results(image, frameNum, bboxes, bbwhs, confidences, class_ids, (sr, sc))
                    putText(disp_image, f'Frame# = {frameNum}, {filename}', row=170, fontScale=0.5)

                    if self.record:
                        # img = cv2.cvtColor(disp_image, code=cv2.COLOR_RGB2BGR)
                        img = resize(disp_image, width=1400)  # not sure why this is needed to stop black screen video
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

            if  stop_frame is not None and stop_frame == frameNum:
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


    def display_results(self, image, frameNum, bboxes, bbwhs, confidences, class_ids, pos):

        (sr, sc) = pos
        # the source image is very large so we reduce it to a more manageable size for display
        # disp_image = cv2.cvtColor(resize(image, width=self.display_width), cv2.COLOR_GRAY2BGR)
        contours, hierarchy = cv2.findContours(getGImages().mask * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        disp_image = resize(image, width=getGImages().small_gray.shape[1])

        for idx in range(len(contours)):
            cv2.drawContours(disp_image, contours, idx, (255,0,0), 1)
        disp_image = resize(disp_image, width=self.display_width)

        angle_inc = sc  * 54.4 / image.shape[1]
        self.heading_angle -= angle_inc
        if self.qgc is not None:
            self.qgc.high_latency2_send(heading=self.heading_angle)
        putText(disp_image, f'{self.heading_angle :.3f}', row=disp_image.shape[0]-20, col=disp_image.shape[1]//2)

        # detected planes
        # argplanes = np.nonzero(class_ids == 3)

        # if len(class_ids) > 0:
        #     class_ids[0] = 3 # make the first one a plane....  JN hack todo fix

        detections = [bb  for bb, d in zip(bbwhs, class_ids) if d == 3]

        for bb in detections:
            wid = bb[2]
            rad_pix = math.radians(54.4 / image.shape[1])
            dist = 40/ (wid * rad_pix)
            heading = (bb[0] - image.shape[1] // 2) * 54.4 / image.shape[1]
            ang = self.heading_angle + heading
            print(f"plane detected at range {dist} {ang} ")
            if self.qgc is not None:
                self.qgc.adsb_vehicle_send('bob', dist, ang, max_step=100)

        # if len(argplanes) > 0:
        #     idx = argplanes.flatten()
        #     ranges = bbwhs[idx][2]
        #     headings = bbwhs[idx][0]
        #
        #     for r in ranges:
        #         print(f"plane detected at range {r}")

        cv2.imshow('blockreduce_mask', resize(getGImages().mask * 255, width=1000))

        cv2.imshow('blockreduce_CMO', cv2.applyColorMap(resize(norm_uint8(getGImages().cmo ), width=1000), cv2.COLORMAP_MAGMA))

        # Tuple of 10 elements representing (frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)
        # remove the ground class
        _bboxes, _confidences, _class_ids = [], [], []
        # for i, (bb,conf,id) in enumerate(zip(bboxes, confidences, class_ids)):
        #     # if i == 0 or (id != 1 and id != 2) :  # cloud and ground
        #     # if conf > 0.05:
        #         _bboxes.append(bb)
        #         _confidences.append(conf)
        #         _class_ids.append(id)

        # tracks = self.tracker.update(_bboxes, _confidences, _class_ids, (sr, sc))
        # _bboxes = [bb.append(conf) for bb, conf in zip(bboxes, confidences)]
        for i, (bb, conf) in enumerate(zip(bboxes, confidences)):
            # if conf > 0.05:
            xyxy_conf = [bb[0]-80, bb[1]-40, bb[0]+80, bb[1]+40, conf]
            _bboxes.append(xyxy_conf)
        # bboxes[:, 2:4] += bboxes[:, 0:2]
        _bboxes = np.array(_bboxes)
        tracks = self.tracker.update(_bboxes, confidences, (sc, sr))

        if self.testpoint is None:
            self.testpoint = (int(disp_image.shape[0] * 0.9), disp_image.shape[1] // 2)

        self.testpoint = (self.testpoint[0] + int(sr * self.display_scale), self.testpoint[1] + int(sc * self.display_scale))
        cv2.circle(disp_image, (self.testpoint[1], self.testpoint[0]), 30, (0, 255, 0), 1)


        # for i in range(len(bboxes)):
        #     bboxes[i] = (bboxes[i] * self.display_scale).astype('int32')

        draw_tracks(disp_image, tracks, dotsize=1, colors=self.model.bbox_colors, display_scale=self.display_scale)

        # self.model.draw_bboxes(disp_image, self.model.bbwhs, None, None, display_scale=self.display_scale, text=False)

        # grab tiles form the src image, accounting for the display scale

        # tiles = make_tile_list(self.model.fullres_img_tile_lst, tracks)
        tiles = []
        tiles = old_make_tile_list(image, tracks)

        if STORE_TILES:
            # logger.warning( " all_tiles.append(tiles)  will hog memory")
            for tl in tiles:
                self.all_tiles.append(tl[0])    # warning this will hog memory
        try:
            tile_img = tile_images(tiles, label=True, colors=self.model.bbox_colors)
        except Exception as e:
            logger.warning(e)
        # tile_img = cv2.cvtColor(resize(tile_img, width=self.display_width), cv2.COLOR_GRAY2BGR)
        tile_img = resize(tile_img, width=self.display_width, inter=cv2.INTER_NEAREST)
        disp_image = np.vstack([tile_img, disp_image])
        return disp_image



if __name__ == '__main__':

    STORE_TILES = False
    STORE_LABELS = False

    RECORD = False
    STOP_FRAME = None

    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.2
    DRAW_BOUNDING_BOXES = True
    USE_GPU = False


    USE_QGC = False
    if USE_QGC:
        from utils.qgcs_connect import ConnectQGC
    # method = 'CentroidKF_Tracker'

    # _tracker = CentroidTracker(max_lost=0, tracker_output_format='visdrone_challenge')
    # _tracker = CentroidTracker(maxDisappeared=5, maxDistance=100)

    # tracker = CentroidKF_Tracker(max_lost=0, tracker_output_format='mot_challenge')
    # tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)
    # tracker = IOUTracker(max_lost=2, iou_threshold=0.5, min_detection_confidence=0.4, max_detection_confidence=0.7,
    #                      tracker_output_format='mot_challenge')

    _tracker = Sort(max_age=5,
                       min_hits=3,
                       iou_threshold=0.2)

    # home = str(Path.home())
    _model = CMO_Peak(confidence_threshold=0.1,
                        labels_path='/home/jn/data/imagenet_class_index.json',
                        # labels_path='/media/jn/0c013c4e-2b1c-491e-8fd8-459de5a36fd8/home/jn/data/imagenet_class_index.json',
                        expected_peak_max=60,
                        peak_min_distance=5,
                        num_peaks=5,
                        maxpool=12,
                        CMO_kernalsize=3,
                        track_boxsize=(80,160),
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
    # path = 'Z:/Day2/seq1/'
    path = home+"/data/large_plane/images"
    # path = home+"/data/ardmore_30Nov21"
    path = home+"/data/Karioitahi_15Jan2022/123MSDCF-35mm"
    # path = home+"/data/Karioitahi_15Jan2022/117MSDCF-28mm(subset)"
    # path = home+"/data/Karioitahi_15Jan2022/117MSDCF-28mm"
    path = home+"/data/Karioitahi_09Feb2022/132MSDCF-28mm-f4"
    # path = home+"/data/Karioitahi_09Feb2022/133MSDCF-28mm-f4"
    # path = home+"/data/Karioitahi_09Feb2022/135MSDCF-60mm-f5.6"
    # path = home+"/data/Karioitahi_09Feb2022/131MSDCF-28mm-f8"
    # path = home+"/data/Karioitahi_09Feb2022/126MSDCF-28mm-f4.5"
    # path = home+"/data/Tairua_15Jan2022/109MSDCF"
    path = home+"/data/orakei_Dec_02/101MSDCF"

    # USE_CAMERA = 'CAM=SONY'
    USE_CAMERA = 'CAM=BASLER'
    # USE_CAMERA = 'FILES'

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

    else :
        loader = il.ImageLoader(path + '/*.JPG', mode='RGB', cvtgray=False, start_frame=0)
    if USE_QGC:
        NZ_START_POS = (-36.9957915731748, 174.91686500754628)
        qgc = ConnectQGC(NZ_START_POS)
    else:
        qgc = None

    main = Main(loader, _model, _tracker, display_width=1500, record=RECORD, path=path, qgc=qgc)

    main.run(wait_timeout=0, heading_angle=-70, stop_frame=STOP_FRAME)

    if STORE_TILES:
        np_tiles = np.asarray(main.all_tiles)
        np.save(path+'np_tiles.npy', np_tiles)

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