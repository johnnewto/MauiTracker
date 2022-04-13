"""
CMO_Peak_JNv1 Object Detector Module.
"""

import numpy as np

import cv2  as cv2

from motrackers.detector import Detector
from motrackers.utils.misc import xyxy2xywh
from motrackers.utils.misc import load_labelsjson
from skimage.feature.peak import peak_local_max
from utils.image_utils import resize, BH_op, get_tile, min_pool, find_sky_2, crop_idx
import  utils.pytorch_utils as ptu
import torch

from motrackers import parameters as pms
import logging
from dataclasses import dataclass

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=pms.LOGGING_LEVEL)
logger = logging.getLogger(__name__)

@dataclass
class Images:
    maxpool:int = 12
    CMO_kernalsize = 3
    full_rgb:np.array = None
    small_rgb:np.array = None
    full_gray:np.array = None
    small_gray:np.array = None
    minpool:np.array = None
    minpool_f:np.array = None
    last_minpool_f:np.array = None
    cmo:np.array = None
    mask:np.array = None
    horizon:np.array = None

    def set(self, image:np.array):
        self.full_rgb = image
        if self.full_rgb.ndim == 3:
            # use this as much faster than cv2.cvtColor(imgrgb, cv2.COLOR_BGR2GRAY) (~24msec for 6K image)
            self.full_gray = self.full_rgb[:,:,1]

        self.minpool = min_pool(self.full_gray, self.maxpool, self.maxpool)
        small_gray = resize(self.full_gray, width=self.minpool.shape[1])
        self.small_rgb = resize(self.full_rgb, width=self.minpool.shape[1])
        self.small_gray = np.zeros_like(self.minpool, dtype='uint8')
        n_rows = min(self.minpool.shape[0], small_gray.shape[0])
        self.small_gray[:n_rows,:] = small_gray[:n_rows,:]
        # self.small_gray = small_gray
        self.minpool_f = np.float32(self.minpool )

    def mask_sky(self):
        self.mask = find_sky_2(self.minpool, threshold=80,  kernal_size=7)
        self.cmo = BH_op(self.minpool, (self.CMO_kernalsize, self.CMO_kernalsize))
        self.cmo[self.mask > 0] = 0

class CMO_Peak(Detector):
    """
    Object Detector Module using intensity peaks
    """

    def __init__(self, confidence_threshold=0.5,
                 labels_path = None,
                 expected_peak_max=60,
                 peak_min_distance=10,
                 num_peaks = 10,
                 maxpool=12,
                 CMO_kernalsize=3,
                 track_boxsize = (160,80),
                 bboxsize=40,
                 draw_bboxes=True,
                 device=None):

        if labels_path is not None:
            object_names = load_labelsjson(labels_path)
        else:
            object_names = {1:'plane', 2:'cloud'}

        # self.scale_factor = 1/255.0
        # self.image_size = (416, 416)
        self.expected_peak_max=expected_peak_max
        self.peak_min_distance = peak_min_distance
        self.num_peaks = num_peaks
        self.maxpool = maxpool
        self.CMO_kernalsize = CMO_kernalsize
        self.bboxsize = bboxsize
        self.image = Images(maxpool=maxpool)
        # self.l_image_sf = None
        # self.cmo : np.ndarray = np.empty([2, 2])
        x = np.arange(-10, 10 + 1, 10)
        y = np.arange(-10, 10 + 1, 10)
        self._xv, self._yv = np.meshgrid(x, y)
        self.device = device
        if self.device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # self._res18_model = ptu.loadModel(device=self.device)
        try:
            self._res18_model = ptu.loadCustomModel(device=self.device)
            self.predictor = ptu.Predict(self._res18_model).to(self.device)
        except Exception as e:
            logger.warning(e)

        super().__init__(object_names, confidence_threshold, draw_bboxes)

    def set_max_pool(self, maxpool=12):
        self.maxpool = maxpool
        self.image.maxpool = self.maxpool
        print(f'Setting maxpool = {self.maxpool}')

    def align(self):
        """ Align this image to the last in order to keep tracking centers accurate"""

        try:
            ((sc, sr), _error) = cv2.phaseCorrelate(self.image.last_minpool_f, self.image.minpool_f)
            self.image.last_minpool_f = self.image.minpool_f
        except:
            self.image.last_minpool_f = self.image.minpool_f
            sc, sr = 0, 0

        return (round(sr*self.maxpool), round(sc*self.maxpool))

    def get_bb(self, img, pos, threshold=0.5):
        """ get the bounding box from the thresholded contour (usually on a CMO image"""
        (r1,c1) = pos
        thresh = int(img[r1, c1] * threshold)
        mask = ((img > thresh) * 255).astype('uint8')
        num_regions, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        # get the second largest region
        if num_regions > 1:
            idx = labels[r1, c1]
            if idx != 0:  # 0 is the background
                bb = stats[idx]
                # bbwh = [(c0 - c1 + bb[0]) * self.maxpool, (r0 - r1 + bb[1]) * self.maxpool, bb[2] * self.maxpool,
                #         bb[3] * self.maxpool]
                bbwh = [bb[0], bb[1], bb[2], bb[3]]
            else:
                bbwh = None
        else:
            bbwh = None  # maybe due to cropping problem
        if bbwh == None:
            bbwh = [mask.shape[1]//2, mask.shape[0]//2, 0, 0]
        return bbwh

    def find_peaks(self):

        # self.image.cmo = BH_op(self.image.minpool, (self.CMO_kernalsize, self.CMO_kernalsize))
        # make it a bit more visible

        # self.mask = mask_horizon(self.images.image_s, threshold=70)
        # self.mask = np.zeros_like(self.image.small_gray, dtype='uint8')
        # self.mask = find_sky_1(self.images.image_s, threshold=80,  kernal_size=7)
        # self.image.mask = find_sky_2(self.image.minpool, threshold=80,  kernal_size=7)
        # self.image.cmo[self.image.mask > 0] = 0


        threshold_abs = self.expected_peak_max*self.confidence_threshold
        _pks = peak_local_max(self.image.cmo,
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

            bs0 = round(self.bboxsize//2)

            lowres_img = get_tile(self.image.small_gray, (r - bs0, c - bs0), (self.bboxsize, self.bboxsize))
            lowres_cmo = get_tile(self.image.cmo, (r - bs0, c - bs0), (self.bboxsize, self.bboxsize))

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
            # find more accurate peak position based on full size image
            r, c = r * self.maxpool, c * self.maxpool

            img = get_tile(self.image.full_rgb, (r - bs, c - bs), (bs * 2, bs * 2))
            fullres_cmo = BH_op(img, (self.CMO_kernalsize*2+1, self.CMO_kernalsize*2+1))
            # l_cmo = self.cmo[r-bs:r+bs, c-bs:c+bs]
            (_r, _c) = np.unravel_index(fullres_cmo.argmax(), fullres_cmo.shape)
            r, c = r-bs+_r, c-bs+_c
            pks.append((r, c))
            img = get_tile(self.image.full_rgb, (r - bs, c - bs), (bs * 2, bs * 2))
            # fullres_cmo = BH_op(img, (self.CMO_kernalsize*2+1, self.CMO_kernalsize*2+1))
            # img = fill_crop(image, (r-bs*2, c-bs*2), (bs*4, bs*4))
            fullres_img = img
            self.fullres_cmo_tile_lst.append(fullres_cmo)
            self.fullres_img_tile_lst.append(fullres_img)


            pk_val = fullres_cmo[bs, bs]
            self.cmo_pk_vals.append(pk_val)
            if pk_val < 1:
                logger.warning(f'Detected Peak Value ({i} : {pk_val}) is very low? Frame {self.frameNum}')

            bbwh = self.get_bb(fullres_cmo, (bs, bs), threshold=0.25)

            # cv2.rectangle(fullres_cmo, (bbwh[0], bbwh[1]), (bbwh[0]+bbwh[2], bbwh[1]+bbwh[3]), (255, 255, 0), 1)
            # cv2.rectangle(fullres_img, (bbwh[0], bbwh[1]), (bbwh[0]+bbwh[2], bbwh[1]+bbwh[3]), (255, 255, 0), 1)
            # draw a line
            # _row = fullres_img.shape[0]//2 -5 # bbwh[1]+bbwh[3]-1
            _row = 20
            _col = bbwh[0]

            wid = bbwh[2]
            fullres_cmo[_row, _col:_col+wid] = (64)
            fullres_img[_row, _col:_col+wid] = (255, 255, 0)


        self.pks = pks
        self.bbwhs = bbwhs

        # gradients are used for filtering out clouds
        self.pk_gradients = []
        for i, (r,c) in enumerate(pks):
            grad = self.image.full_gray[r+self._yv, c+self._xv].astype(np.int32)
            grad = grad - grad[1, 1]
            grad[1, 1] = grad.max()
            self.pk_gradients.append(grad)

        return self.pks, self.bbwhs

    def test_foward(self):
        pass

    def classify(self, detections, image, pk_vals, pk_gradients, scale, filterClassID):
        # if filterClassID is None:
        #     filterClassID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        fullres_img_tile_lst, bboxes, confidences, class_ids = [], [], [], []
        for i, detection in enumerate(detections):
            # determine if cloud or not
            confidence = pk_vals[i] / self.expected_peak_max

            if pk_gradients[i].min() < 20 and pk_gradients[i].min() < pk_gradients[i].max() // 2:  # todo fix magic  number for cloud threshold
                class_id = 2  # cloud
            else:
                class_id = 1  # plane pk_gradientsor bird

            if confidence > self.confidence_threshold and class_id in filterClassID:
                bs = self.bboxsize//2
                # bbox = detection[3:7] * np.array([self.width, self.height, self.width, self.height])
                row, col = int(detection[0] * scale), int(detection[1] * scale)
                bbox = (col-bs, row-bs, col+bs, row+bs)
                bboxes.append(bbox)
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
                fullres_img_tile_lst.append(image[row - bs:row + bs, col - bs:col + bs])

        return bboxes, confidences, class_ids, fullres_img_tile_lst

    def old_classifyNN(self, detections, image):

        nn_tile_list, bboxes, confidences, class_ids = [], [], [], []
        for i, detection in enumerate(detections):
            # determine if cloud or not
            bs = self.bboxsize//2
            # bbox = detection[3:7] * np.array([self.width, self.height, self.width, self.height])
            # row, col = int(detection[0] * scale), int(detection[1] * scale)
            row, col = detection
            row, col = crop_idx(row, col, bs, image.shape)
            bbox = (col-bs, row-bs, col+bs, row+bs)
            bboxes.append(bbox)
            # (col, row, h, w) = [round(v / scale) for v in (col, row, h, w)]
            # bbox = [round(v / scale) for v in bbox]
            # row = image.shape[0] - h if row >= image.shape[0] - h else row
            # col = image.shape[1] - w if col >= image.shape[1] - w else col
            img = image[row-bs:row+bs, col-bs:col+bs]
            assert img.shape[:2] == (self.bboxsize, self.bboxsize)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            nn_tile_list.append(img)  # tuple with (numpy, labelID, defect pos. frameID)

        batch = ptu.images2batch(nn_tile_list).to(self.device)
        y_pred = self.predictor(batch)
        cat, conf = self.predictor.conf(y_pred)
        class_ids = cat.detach().cpu().numpy().tolist()
        confidences = conf.detach().cpu().numpy().tolist()

        return bboxes, confidences, class_ids, nn_tile_list

        # y_pred = self.predictor(batch)
        # cat, conf = self.predictor.conf(y_pred)
        # print(cat)
        # print(conf)

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

    # def mask_sky(self):
    #     self.image.mask = find_sky_2(self.image.minpool, threshold=80,  kernal_size=7)
    #     self.image.cmo = BH_op(self.image.minpool, (self.CMO_kernalsize, self.CMO_kernalsize))
    #     self.image.cmo[self.image.mask > 0] = 0

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
            (self.height, self.width) = self.image.full_rgb.shape[:2]

        (sr, sc) = self.align()
        detections, bbwhs = self.find_peaks()
        centers = [ (bb[1]+bb[3]//2, bb[0]+bb[2]//2)  for bb in bbwhs]
        bboxes = []
        for detection in detections:
            bs = self.bboxsize//2
            row, col = detection
            row, col = crop_idx(row, col, bs, self.image.full_rgb.shape)
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