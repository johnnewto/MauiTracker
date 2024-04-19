"""
This module contains the implementation of the CMO_Peak class, which is an object detector module using intensity peaks.
"""

import numpy as np
import cv2 as cv2
from utils.detector import Detector
from motrackers.utils.misc import xyxy2xywh
from motrackers.utils.misc import load_labelsjson
from skimage.feature.peak import peak_local_max
from utils.image_utils import resize, BH_op, get_tile, min_pool, crop_idx
from utils.g_images import *
from utils.horizon import find_sky_2
import utils.pytorch_utils as ptu
import torch
from utils import parameters as pms
import logging

logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=pms.LOGGING_LEVEL)
logger = logging.getLogger(__name__)


class CMO_Peak(Detector):
    """
    Object Detector Module using intensity peaks
    """

    def __init__(self, confidence_threshold=0.5,
                 labels_path=None,
                 expected_peak_max=60,
                 peak_min_distance=10,
                 num_peaks=10,
                 maxpool=12,
                 CMO_kernalsize=3,
                 track_boxsize=(160, 80),
                 bboxsize=40,
                 draw_bboxes=True,
                 device=None):
        """
        Initializes the CMO_Peak object detector module.

        Args:
            confidence_threshold (float): Confidence threshold for object detection.
            labels_path (str): Path to the labels JSON file.
            expected_peak_max (int): Maximum expected peak value.
            peak_min_distance (int): Minimum distance between peaks.
            num_peaks (int): Number of peaks to detect.
            maxpool (int): Maximum pooling size.
            CMO_kernalsize (int): Kernel size for CMO operation.
            track_boxsize (tuple): Size of the tracking box.
            bboxsize (int): Size of the bounding box.
            draw_bboxes (bool): Whether to draw bounding boxes on the image.
            device (str): Device to use for computation (e.g., "cuda:0" or "cpu").
        """
        if labels_path is not None:
            object_names = load_labelsjson(labels_path)
        else:
            object_names = {1: 'plane', 2: 'cloud'}

        self.expected_peak_max = expected_peak_max
        self.peak_min_distance = peak_min_distance
        self.num_peaks = num_peaks
        self.maxpool = maxpool
        self.CMO_kernalsize = CMO_kernalsize
        self.bboxsize = bboxsize

        x = np.arange(-10, 10 + 1, 10)
        y = np.arange(-10, 10 + 1, 10)
        self._xv, self._yv = np.meshgrid(x, y)
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        try:
            self._res18_model = ptu.loadCustomModel(device=self.device)
            self.predictor = ptu.Predict(self._res18_model).to(self.device)
        except Exception as e:
            logger.warning(e)

        super().__init__(object_names, confidence_threshold, draw_bboxes)

    def set_max_pool(self, maxpool=12):
        """
        Sets the maximum pooling size.

        Args:
            maxpool (int): Maximum pooling size.
        """
        self.maxpool = maxpool
        getGImages().maxpool = self.maxpool
        print(f'Setting maxpool = {self.maxpool}')

    def align(self):
        """
        Aligns the image to the last in order to keep tracking centers accurate.

        Returns:
            tuple: Tuple containing the aligned coordinates (row, column).
        """
        try:
            ((sc, sr), _error) = cv2.phaseCorrelate(getGImages().last_minpool_f, getGImages().minpool_f)
            getGImages().last_minpool_f = getGImages().minpool_f
        except:
            getGImages().last_minpool_f = getGImages().minpool_f
            sc, sr = 0, 0

        return (round(sr * self.maxpool), round(sc * self.maxpool))

    def get_bb(self, img, pos, threshold=0.5):
        """
        Gets the bounding box from the thresholded contour (usually on a CMO image).

        Args:
            img (numpy.ndarray): Input image.
            pos (tuple): Position of the peak.
            threshold (float): Threshold value.

        Returns:
            list: List containing the bounding box coordinates [x, y, width, height].
        """
        (r1, c1) = pos
        thresh = int(img[r1, c1] * threshold)
        mask = ((img > thresh) * 255).astype('uint8')
        num_regions, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_regions > 1:
            idx = labels[r1, c1]
            if idx != 0:
                bb = stats[idx]
                bbwh = [bb[0], bb[1], bb[2], bb[3]]
            else:
                bbwh = None
        else:
            bbwh = None
        if bbwh is None:
            bbwh = [mask.shape[1] // 2, mask.shape[0] // 2, 0, 0]
        return bbwh

    def find_peaks(self):
        """
        Finds the intensity peaks in the image.

        Returns:
            tuple: Tuple containing the peak positions and bounding boxes.
        """
        threshold_abs = self.expected_peak_max * self.confidence_threshold
        _pks = peak_local_max(getGImages().cmo,
                              min_distance=self.peak_min_distance,
                              threshold_abs=threshold_abs,
                              num_peaks=self.num_peaks)

        self.fullres_cmo_tile_lst = []
        self.fullres_img_tile_lst = []
        self.lowres_cmo_tile_lst = []
        self.lowres_img_tile_lst = []
        pks = []
        bbwhs = []
        self.cmo_pk_vals = []

        # get low res and full res peaks
        # gather all the tiles centered on the peak positions
        for i, (r, c) in enumerate(_pks):
            bs0 = round(self.bboxsize // 2)
            lowres_img = get_tile(getGImages().small_gray, (r - bs0, c - bs0), (self.bboxsize, self.bboxsize))
            lowres_cmo = get_tile(getGImages().cmo, (r - bs0, c - bs0), (self.bboxsize, self.bboxsize))
            self.lowres_img_tile_lst.append(lowres_img)
            self.lowres_cmo_tile_lst.append(lowres_cmo)
            bbwh = self.get_bb(lowres_cmo, (bs0, bs0))
            if bbwh is not None:
                bbwh = [(c - bs0 + bbwh[0]) * self.maxpool, (r - bs0 + bbwh[1]) * self.maxpool, bbwh[2] * self.maxpool,
                        bbwh[3] * self.maxpool]
                bbwhs.append(bbwh)
            else:
                logger.error("bbwh could not be created")
            bs = self.bboxsize
            r, c = r * self.maxpool, c * self.maxpool
            img = get_tile(getGImages().full_rgb, (r - bs, c - bs), (bs * 2, bs * 2))
            fullres_cmo = BH_op(img, (self.CMO_kernalsize * 2 + 1, self.CMO_kernalsize * 2 + 1))
            (_r, _c) = np.unravel_index(fullres_cmo.argmax(), fullres_cmo.shape)
            r, c = r - bs + _r, c - bs + _c
            pks.append((r, c))
            img = get_tile(getGImages().full_rgb, (r - bs, c - bs), (bs * 2, bs * 2))
            fullres_img = img
            self.fullres_cmo_tile_lst.append(fullres_cmo)
            self.fullres_img_tile_lst.append(fullres_img)
            pk_val = fullres_cmo[bs, bs]
            self.cmo_pk_vals.append(pk_val)
            if pk_val < 1:
                logger.warning(f'Detected Peak Value ({i} : {pk_val}) is very low? Frame {self.frameNum}')
            bbwh = self.get_bb(fullres_cmo, (bs, bs), threshold=0.25)
            _row = 20
            _col = bbwh[0]
            wid = bbwh[2]
            fullres_cmo[_row, _col:_col + wid] = (64)
            fullres_img[_row, _col:_col + wid] = (255, 255, 0)

        self.pks = pks
        self.bbwhs = bbwhs
        self.pk_gradients = []
        for i, (r, c) in enumerate(pks):
            grad = getGImages().full_gray[r + self._yv, c + self._xv].astype(np.int32)
            grad = grad - grad[1, 1]
            grad[1, 1] = grad.max()
            self.pk_gradients.append(grad)

        return self.pks, self.bbwhs

    def test_foward(self):
        """
        Test forward method.
        """
        pass

    def classify(self, detections, image, pk_vals, pk_gradients, scale, filterClassID):
        """
        Classifies the detections based on peak values and gradients.

        Args:
            detections (list): List of detections.
            image (numpy.ndarray): Input image.
            pk_vals (list): List of peak values.
            pk_gradients (list): List of peak gradients.
            scale (float): Scaling factor.
            filterClassID (list): List of class IDs to filter.

        Returns:
            tuple: Tuple containing the bounding boxes, confidences, class IDs, and full resolution image tiles.
        """
        fullres_img_tile_lst, bboxes, confidences, class_ids = [], [], [], []
        for i, detection in enumerate(detections):
            confidence = pk_vals[i] / self.expected_peak_max
            if pk_gradients[i].min() < 20 and pk_gradients[i].min() < pk_gradients[i].max() // 2:
                class_id = 2  # cloud
            else:
                class_id = 1  # plane or bird
            if confidence > self.confidence_threshold and class_id in filterClassID:
                bs = self.bboxsize // 2
                row, col = int(detection[0] * scale), int(detection[1] * scale)
                bbox = (col - bs, row - bs, col + bs, row + bs)
                bboxes.append(bbox)
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
                fullres_img_tile_lst.append(image[row - bs:row + bs, col - bs:col + bs])

        return bboxes, confidences, class_ids, fullres_img_tile_lst

 
    def classifyNN(self, tile_list):
        """ classify the full res colour tiles """
        batch = ptu.images2batch(tile_list).to(self.device)
        y_pred = self.predictor(batch)
        cat, conf = self.predictor.conf(y_pred)
        class_ids = cat.detach().cpu().numpy().tolist()
        confidences = conf.detach().cpu().numpy().tolist()

        return confidences, class_ids

        # y_pred = self.predictor(batch)
        # cat, conf = self.predictor.conf(y_pred)
        # print(cat)
        # print(conf)



    def detect(self, scale=1, filterClassID=None, frameNum=None):
        """
        Detect objects in the input image.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            tuple: Tuple containing the following elements:
                - bboxes (numpy.ndarray): Bounding boxes with shape (n, 4) containing constant sized objects with each row as `(xmin, ymin, width, height)`.
                - bbwh (numpy.ndarray): Bounding boxes with shape (n, 4) containing accurate detected objects with each row as `(xmin, ymin, width, height)`.
                - confidences (numpy.ndarray): Confidence or detection probabilities if the detected objects with shape (n,).
                - class_ids (numpy.ndarray): Class_ids or label_ids of detected objects with shape (n, 4)
        """
        if filterClassID is None:
            filterClassID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        self.frameNum = frameNum
        if self.width is None or self.height is None:
            (self.height, self.width) = getGImages().full_rgb.shape[:2]

        (sr, sc) = self.align()
        detections, bbwhs = self.find_peaks()
        centers = [ (bb[1]+bb[3]//2, bb[0]+bb[2]//2)  for bb in bbwhs]
        bboxes = []
        for detection in detections:
            bs = self.bboxsize//2
            row, col = detection
            row, col = crop_idx(row, col, bs, getGImages().full_rgb.shape)
            bbox = (col-bs, row-bs, col+bs, row+bs)
            bboxes.append(bbox)

        try:
            confidences, class_ids = self.classifyNN(self.fullres_img_tile_lst)
        except Exception as e:
            # self.classify(detections, image, pk_vals, pk_gradients, scale, filterClassID)
            logger.error(e)
            # confidences = [1] * len(self.fullres_img_tile_lst)
            confidences = [cmo_pk_val / 255.0 for cmo_pk_val in self.cmo_pk_vals]
            class_ids = [0] * len(self.fullres_img_tile_lst)

        # for bbwh, tile in zip(self.bbwhs, self.fullres_img_tile_lst):
        #     if bbwh is not None:
        #         r,c = tile.shape[0]//2, tile.shape[1]//2
        #         w2, h2 = bbwh[2]//2, bbwh[3]//2
        #         pos = (c-w2, r-h2)
        #         end = (c+w2, r+h2)
        #         cv2.rectangle(tile, pos, end, (255, 255, 0), 1)
        bboxes = xyxy2xywh(np.array(bboxes)).tolist()
        return bboxes, bbwhs, confidences, class_ids, (sr, sc)

        # if len(confidences):
        #     bboxes = xyxy2xywh(np.array(bboxes)).tolist()
        #     # class_ids = np.array(class_ids).astype('int')
        #     # return np.array(bboxes), np.array(bbwhs), np.array(confidences), class_ids, (sr, sc)
        #     return bboxes, bbwhs, confidences, class_ids, (sr, sc)
        # else:
        #     return [], [], [], [], (sr, sc)
            # return np.array([]), np.array([]), np.array([]), np.array([]), (sr, sc)

    def draw_bboxes(self, image, bboxes, confidences, class_ids, display_scale=None, text=True):
        """
        Draw the bounding boxes about detected objects in the image.

        Args:
            image (numpy.ndarray): Image or video frame.
            bboxes (numpy.ndarray): Bounding boxes pixel coordinates as (xmin, ymin, width, height)
            confidences (numpy.ndarray): Detection confidence or detection probability.
            class_ids (numpy.ndarray): Array containing class ids (aka label ids) of each detected object.
            display_scale: ratio of bbbox coordinates to the actual display coords

        Returns:
            numpy.ndarray: image with the bounding boxes drawn on it.
        """
        if confidences is None:
            confidences = [1.0 for bb in bboxes]
        if class_ids is None:
            class_ids = [i for i, bb in enumerate(bboxes)]

        for bb, conf, cid in zip(bboxes, confidences, class_ids):
            clr = [int(c) for c in self.bbox_colors[cid]]
            if display_scale is not None:
                for i in range(len(bb)):
                    bb[i] = int(bb[i] * display_scale)
                # bb[0], bb[1] = int(bb[0]*display_scale), int(bb[1]*display_scale)
            cv2.rectangle(image, (bb[0], bb[1] ), (bb[0] + bb[2], bb[1] + bb[3]), clr, 1)
            if text:
                label = f"{self.object_names[cid]} : {conf:.2f}"
                (label_width, label_height), baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y_label = max(bb[1], label_height)
                cv2.rectangle(image, (bb[0], y_label - label_height), (bb[0] + label_width, y_label + baseLine),
                             (255, 255, 255), cv2.FILLED)
                cv2.putText(image, label, (bb[0], y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1)
        return image


        # for bb, conf, cid in zip(bboxes, confidences, class_ids):
        #     clr = [int(c) for c in self.bbox_colors[cid]]
        #     cv.rectangle(image, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), clr, 1)
        #     label = "{}:{:.4f}".format(self.object_names[cid], conf)
        #     (label_width, label_height), baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        #     y_label = max(bb[1], label_height)
        #     cv.rectangle(image, (bb[0], y_label - label_height), (bb[0] + label_width, y_label + baseLine),
        #                  (255, 255, 255), cv.FILLED)
        #     cv.putText(image, label, (bb[0], y_label), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1)
        # return image