__all__ = [ 'get_project_root', 'near_mask', 'mask_horizon_1', 'mask_horizon_2', 'find_sky_1', 'find_sky_2', 'set_horizon']

import cv2 as cv2
import numpy as np
from utils.show_images import cv2_img_show
from utils import parameters as pms

import time, sys

if sys.platform == "win32":
    # On Windows, the best timer is time.clock()
    default_timer = time.perf_counter
else:
    # On most other platforms the best timer is time.time()
    default_timer = time.time

import logging
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=pms.LOGGING_LEVEL)
logger = logging.getLogger(__name__)

from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def mask_horizon_1(image, shrink=None, calcOtsu_period=10, kernal_size=5, factor=0.7, threshold=None):
    try:
        mask_horizon_1.count += 1
    except AttributeError:
        mask_horizon_1.count = 0

    if shrink is not None:
        image = resize(image, width=image.shape[1] // shrink)

    mask_horizon_1.threshold = threshold
    if threshold is None and mask_horizon_1.count % calcOtsu_period == 0:
        (mask_horizon_1.threshold, mask) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        mask_horizon_1.threshold = mask_horizon_1.threshold*factor
        print(f'Count {mask_horizon_1.count} Threshold {mask_horizon_1.threshold}')

    (T, mask) = cv2.threshold(image, mask_horizon_1.threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernal_size, kernal_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    return mask

def mask_horizon_2(image, shrink=None, calcOtsu_period=10, kernal_size=5, factor=0.7, threshold=None):
    try:
        mask_horizon_2.count += 1
    except AttributeError:
        mask_horizon_2.count = 0

    mask = cv2.Canny(image, threshold1=50, threshold2=150, apertureSize=3)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)  # dilation give more candidates for Hough lines
    # cv2_img_show('Canny1', edges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return mask

# edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)

def get_neighbour(labels, pos, end):
    """ get a neighouring region: useful for removing small regions """
    labels_shape = np.array(labels.shape)
    # get  outside points at the diagional points
    pos = np.clip(pos, a_min=0, a_max=labels_shape-1)
    end = np.clip(end, a_min=0, a_max=labels_shape-1)
    pos1 = np.clip(pos-1, a_min=0, a_max=labels_shape-1)
    end1 = np.clip(end+1, a_min=0, a_max=labels_shape-1)
    # if clipped box point is outside ( = a neighbour) then return the neighbour label
    if np.any(pos != pos1):
        return labels[pos1[0],pos1[1]]
    if np.any(end != end1):
        return labels[end1[0],end1[1]]

    return None


def find_sky_1(img, threshold=None, kernal_size=5):
    mask = mask_horizon_1(img, threshold=threshold, kernal_size=kernal_size)
    # mask = 255 - mask
    num_regions, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    area_sort = np.argsort(stats[:, -1])
    # choose the region that is the largest brightest
    brightest = 0
    sky_idx = -1

    # get bright large area
    for i in range(min(num_regions, 3)):
        idx = area_sort[-(i+1)]
        b = np.mean(img[labels==idx])
        area_ratio = stats[idx][4]/(mask.shape[0]*mask.shape[1])
        if b > brightest and area_ratio > 0.25:
            brightest = b
            sky_idx = idx

    assert sky_idx > -1

    # prune regions
    for idx in range(num_regions):
        area_ratio = stats[idx][4] / (mask.shape[0] * mask.shape[1])
        pos = stats[idx][1::-1]  # (col1, col0)
        end = pos + stats[idx][3:1:-1]
        left_side = pos[1]
        right_side = end[1]
        width_ratio = (right_side - left_side) / img.shape[1]
        top_side = pos[0]
        bot_side = end[0]

        # remove small regions ( area < 5%) that are not adjacent to bottom or left or right sides
        if area_ratio < 0.05 and bot_side < img.shape[0] and left_side > 0 and right_side < img.shape[1]:
            neigh_idx = get_neighbour(labels, pos, end)
            # print(idx, neigh_idx)
            labels[labels == idx] = neigh_idx

        # # merge in sky_index wide regions ( width > 5%) that are adjacent to top
        # if width_ratio > 0.05 and top_side == 0 :
        #     labels[labels == idx] = sky_idx


    # merge in sky_index large regions ( area > 10%) that are adjacent to top
    # for seascapes
    #
    # for idx in range(num_regions):
    #     area_ratio = stats[idx][4] / (mask.shape[0] * mask.shape[1])
    #     pos = stats[idx][1::-1]  # (col1, col0)
    #     end = pos + stats[idx][3:1:-1]
    #     left_side = pos[0]
    #     right_side = end[0]
    #     if area_ratio > 0.1 and (left_side == 0 or right_side == img.shape[1]):
    #         labels[labels == idx] = sky_idx


    new_mask = np.ones_like(mask)
    new_mask[labels == sky_idx] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernal_size*2+1, kernal_size*2+1))   # needs to be odd
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernal_size//2+1, kernal_size//2+1))
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_DILATE, kernel, iterations=1)
    # new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_DILATE, kernel)
    return new_mask

def find_sky_2(gray_img_s, threshold=None, kernal_size=5):
    """ optimised for marine images"""
    edges = cv2.Canny(gray_img_s, threshold1=50, threshold2=255, apertureSize=3)
    kernel = np.ones((3, 5), 'uint8')
    edges = cv2.dilate(edges, kernel, iterations=1)  # < --- Added a dilate, check link I provided
    # kernel = np.ones((3, 5), 'uint8')
    # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=5)
    # kernel = np.ones((5, 5), 'uint8')
    # edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)
    # cv2_img_show('find_sky_2-canny', edges)
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
        brightave1 = np.mean(gray_img_s[labels == idx])
        area_ratio = stats[idx][4] / (mask.shape[0] * mask.shape[1])
        if brightave1 > brightest and area_ratio > 0.25:
            brightest = brightave1
            sky_idx = idx

    assert sky_idx > -1
    labels[labels != sky_idx] = 255
    labels[labels == sky_idx] = 0
    labels = labels.astype('uint8')
    kernel = np.ones((5, 5), 'uint8')
    # edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)

    cv2_img_show('find_sky_2-new_mask', labels)

    num_regions, labels, stats, centroids = cv2.connectedComponentsWithStats(labels)

    for idx in range(num_regions):
        area_ratio = stats[idx][4] / (mask.shape[0] * mask.shape[1])
        pos = stats[idx][1::-1]  # (col1, col0)
        end = pos + stats[idx][3:1:-1]
        left_side = pos[1]
        right_side = end[1]
        # top_side = pos[0]
        bot_side = end[0]
        # brightave2 = np.mean(gray_img_s[labels == idx])

        # remove small regions ( area < 10%) that are not adjacent to bottom or left or right sides
        if area_ratio < 0.1:
            if not (bot_side == gray_img_s.shape[0] or left_side == 0 or right_side == gray_img_s.shape[1]):    
                labels[labels == idx] = 0

            # brightmin = np.min(gray_img_s[labels == idx])
            # # print(brightmin, brightest)
            # if brightmin > 50:
            #     labels[labels == idx] = 0

    labels[labels > 0] = 255
    labels = labels.astype('uint8')
    cv2_img_show('find_sky_2-labels pruned', labels)
    return labels

def o_find_sky(img, threshold=None, kernal_size=5):
    mask = mask_horizon_1(img, threshold=threshold, kernal_size=kernal_size)
    # mask = 255 - mask
    num_regions, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    area_sort = np.argsort(stats[:, -1])
    # choose the region that is the largest brightest
    brightest = 0
    b_idx = -1

    # get bright large area
    for i in range(min(num_regions, 3)):
        idx = area_sort[-(i+1)]
        b = np.mean(img[labels==idx])
        area_ratio = stats[idx][4]/(mask.shape[0]*mask.shape[1])
        if b > brightest and area_ratio > 0.25:
            brightest = b
            b_idx = idx

    assert b_idx > -1

    # remove small regions
    for idx in range(num_regions):
        area_ratio = stats[idx][4] / (mask.shape[0] * mask.shape[1])
        if area_ratio < 0.25:
            pos = stats[idx][1::-1]  # (col1, col0)
            end = pos + stats[idx][3:1:-1]
            neigh_idx = get_neighbour(labels, pos, end)
            # print(idx, neigh_idx)
            labels[labels == idx] = neigh_idx

    new_mask = np.ones_like(mask)
    new_mask[labels==b_idx] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernal_size//2, kernal_size//2))
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_DILATE, kernel, iterations=3)
    # new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_DILATE, kernel)
    return new_mask


def old_find_sky(img, threshold=None, kernal_size=10):
    mask = mask_horizon(img, threshold=threshold, kernal_size=kernal_size)
    mask = 255 - mask
    num_regions, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    area_sort = np.argsort(stats[:, -1])
    # choose the region that is the largest brightest
    brightest = 0
    b_idx = -1
    for i in range(min(num_regions, 3)):
        idx = area_sort[-(i+1)]
        b = np.mean(img[labels==idx])
        area_ratio = stats[idx][4]/(mask.shape[0]*mask.shape[1])
        if b > brightest and area_ratio > 0.25:
            brightest = b
            b_idx = idx

    assert b_idx > -1
    new_mask = np.ones_like(mask)
    new_mask[labels==b_idx] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernal_size, kernal_size))
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernal_size//2, kernal_size//2))
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_DILATE, kernel, iterations=3)
    # new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_DILATE, kernel)
    return new_mask

def near_mask(pnt, mask, dist=10):
    try:
        xv, yv = near_mask.xv, near_mask.yv
    except AttributeError:
        x = np.arange(-dist, dist + 1, dist)
        near_mask.xv, near_mask.yv = np.meshgrid(x, x)
        xv, yv = near_mask.xv, near_mask.yv

    r,c = pnt
    _mask = mask[r+yv, c+xv].astype(np.int32)
    return 1 in _mask


def set_horizon(gray_img_s):

    # gray_img_s = getGImages().small_gray.copy()

    edges = cv2.Canny(gray_img_s, threshold1=50, threshold2=250, apertureSize=5)
    # cv2_img_show('Canny1`', edges)
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
    labels[labels != sky_idx] = 255
    labels[labels == sky_idx] = 0
    labels = labels.astype('uint8')
    kernel = np.ones((5, 5), np.uint8)
    labels = cv2.erode(labels, kernel, iterations=1)  # < --- Added a dilate, check link I provided

    return labels
    # kernel = np.ones((5, 5), 'uint8')
    # labels = cv2.morphologyEx(labels, cv2.MORPH_CLOSE, kernel, iterations=5)
    # cv2_img_show('labels', labels)
    # getGImages().horizon = labels



if __name__ == '__main__':
    from utils.show_images import putText

    (rows, cols) = (2000, 3000)
    center = (2750, 4350)
    (_r, _c) = (center[0]-rows//2, center[1]-cols//2)
    crop = [_r, _r + rows, _c, _c + cols]
    home = str(Path.home())
    images = ImageLoader(home+"/data/large_plane/images.npy", crop=crop, scale=0.1, color='Gray')
    wait_timeout = 100
    for img, i in images:
        # cmo =  update(cmo)
        # img = next(images)
        img = resize(img, width=500)
        putText(img, f'Frame = {i}, fontScale=0.5')
        cv2.imshow('image',  img)
        k = cv2.waitKey(wait_timeout)
        if k == ord('q') or k == 27:
            break
        if k == ord(' '):
            wait_timeout = 0
        if k == ord('d'):
            wait_timeout = 0
            images.direction_fwd = not images.direction_fwd
        if k == ord('g'):
            wait_timeout = 100
        if k == ord('r'):
            # change direction
            wait_timeout = 0
            images.restep = True

    cv2.waitKey(1000)
    cv2.destroyAllWindows()

