"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    https://github.com/abewley/sort
"""

from __future__ import print_function

import numpy as np
from filterpy.kalman import KalmanFilter
import cv2

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)


def convert_bbox_to_z(bbox):
    """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

# def convert_xywh2xyxy(xywh, score=None):
#     """
#   Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
#     [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
#   """
#
#     if len(xywh.shape) == 2:
#         x = xywh[:, 0] + xywh[:, 2]
#         y = xywh[:, 1] + xywh[:, 3]
#         xyxy = np.concatenate((xywh[:, 0:2], x[:, None], y[:, None]), axis=1).astype('int')
#         return xyxy
#     if len(xywh.shape) == 1:
#         x, y, w, h = xywh
#         xr = x + w
#         yb = y + h
#         return np.array([x, y, xr, yb]).astype('int')

class KalmanBoxTracker(object):
    """
  This class represents the internal state of individual tracked objects observed as bbox.
  Your job as a designer will be to design the state (x,P), the process (F, Q), the measurement (z, R),
  and the measurement function (H).
  If the system has control inputs, such as a robot, you will also design B and u.
  """
    count = 0

    def __init__(self, bbox):
        """
    Initialises a tracker using initial bounding box.
    """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # self.s = Saver(self.kf)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.score = 0

    def update(self, bbox, score=0):
        """
    Updates the state vector with observed bbox.
    """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.score = score

    def predict(self):
        """
    Advances the state vector and returns the predicted bounding box estimate.
    """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
    Returns the current bounding box estimate.
    """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
    Sets key parameters for SORT
    """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5)), scores=np.zeros(5), shift=(0,0)):
        """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        shift = np.array([[shift[0]], [shift[1]]])
        for t, trk in enumerate(trks):
            self.trackers[t].kf.x[:2] += shift
            self.trackers[t].kf.x_post[:2] += shift
            self.trackers[t].kf.x_prior[:2] += shift
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        try:
            for m in matched:
                self.trackers[m[1]].update(dets[m[0], :], scores[m[0]])
        except:
            pass
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1, trk.score])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 6))



class Animate:
    def __init__(s, dt=0.1, num=5, size=600):
        s.dt = dt
        s.n = num
        s.size = size
        s.particles = np.zeros(s.n, dtype=[("position", 'float', 2),
                                           ("velocity", 'float', 2),
                                           ("force", 'float', 2),
                                           ("value", 'int', ),
                                           ("del_value", 'int', )])


        s.particles["position"] = np.random.uniform(0, 1, (s.n, 2))
        s.particles["position"] = 0.5 * np.ones((s.n, 2))
        s.particles["position"][0] = 0.5 * np.ones((1, 2))

        # s.particles["velocity"] = np.zeros((s.n, 2))
        s.particles["velocity"] = np.random.randint(-2, 2, (s.n, 2)) * s.dt
        # s.particles["velocity"][-1] = 0 * np.ones((1, 2))

        s.particles["value"] = 128 * np.ones(s.n, int)
        s.particles["del_value"] = 5 * np.ones(s.n, int)
        s.image = np.zeros((size, size), dtype=np.uint8) + 1

    def update(s, shift=(0,0)):
        # if frame_number < 5:
        #     s.particles["force"] = np.round(np.random.uniform(-2, 2., (s.n, 2)))
        #     s.particles["velocity"] = s.particles["velocity"] + s.particles["force"] * s.dt
        #     s.particles["velocity"] = np.clip(s.particles["velocity"], -0.1, 0.1)
        #     s.particles["velocity"] = np.random.randint(-1,1,(s.n, 2))*0.1
        #     s.particles["velocity"][0] = 0 * np.ones((1, 2))
        #     print( s.particles["velocity"] )

        s.particles["del_value"] = np.random.uniform(-2, 2., (s.n))
        shift = np.array([shift[0], shift[1]])/s.size
        s.particles["position"] += shift
        s.particles["position"] = s.particles["position"] + s.particles["velocity"] * s.dt
        s.particles["position"] = s.particles["position"] % 1
        # s.particles["value"] = s.particles["value"] + (s.particles["del_value"] * 5).astype(int)
        # s.particles["value"] = np.clip(s.particles["value"], 10,255)


        s.image = np.zeros((s.size, s.size), dtype=np.uint8) + 1
        for particle in s.particles:
            r, c = particle["position"]
            # r = min(int(r * s.size), s.size-5)
            # c = min(int(c * s.size), s.size-5)

            # s.image[r:r + 5, c:c + 5] = 255
            color = int(particle["value"])
            r, c = int(r * s.size), int(c* s.size)

            s.image = cv2.circle(s.image, (c, r), 1, color, -1)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from numpy.random import randn
    from filterpy.common import Saver
    from filterpy.stats import plot_covariance
    from book_plots import plot_filter
    from book_plots import plot_measurements
    from book_plots import plot_residual_limits, set_labels

    def plot_residuals(xs, data, col, title, y_label, stds=1):
        res = xs[:, col] - data.x[:, col].squeeze()
        plt.plot(res)
        plot_residual_limits(data.P[:, col, col], stds)
        set_labels(title, 'frame', y_label)

    def main():

        NUM_PNTS = 2

        total_time = 0.0
        total_frames = 0
        colours = np.random.rand(32, 3)  # used only for display


        from skimage.feature.peak import peak_local_max
        animate = Animate(0.1, num=NUM_PNTS, size=500)
        animate.particles["velocity"] = np.ones((1,2)) * 0.1
        animate.particles["velocity"][0] = np.zeros((1, 2))
        animate.particles["position"][0] = np.ones((1,2)) * 0.75
        mot_tracker = Sort(max_age=3,
                           min_hits=3,
                           iou_threshold=0.3)

        xs = []
        zs = []
        for i in range(180):
            animate.update(shift=(0,0))

            image = cv2.cvtColor(animate.image.copy(), cv2.COLOR_GRAY2BGR)
            # pks = peak_local_max(cmo, min_distance=8, threshold_abs=10, num_peaks=10)
            pks = peak_local_max(animate.image, min_distance=20, threshold_abs=10, num_peaks=NUM_PNTS*2)
            bboxes = [np.array([c,r, 10,10], 'int32') for (r, c) in pks]
            confidences = [animate.image[r,c]/255 for (r,c) in pks]
            class_ids = [1]*pks.shape[0]
            try:
                xs.append([pks[0,1], pks[0,0]])
                pks = pks + randn(pks.shape[0], pks.shape[1]) * 0.2

                dets = np.array([convert_x_to_bbox([c, r, 900, 1], 0.9).squeeze() for (r, c) in pks])
                if i == 0:
                    dets_1 = dets.copy()
                trackers = mot_tracker.update(dets, shift=(0,0))
                try:
                    z = convert_bbox_to_z(dets[0])
                    zs.append(z)
                except:
                    pass

                for d in trackers:
                    color = colours[int(d[4]) % 32, :] * 255
                    color = (int(color[0]), int(color[1]), int(color[2]))
                    rect = d[:4].astype('int32')
                    image = cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)
            except:
                pass

            cv2.imshow('frame', image)

            k = cv2.waitKey(10)
            if k == ord('q') or k == 27:
                break

        # run batch filter again on zs
        xs = np.array(xs)
        zs = np.array(zs)
        # initialise filter again
        bb = dets_1.squeeze()
        kf = KalmanBoxTracker(bb).kf
        s = Saver(kf)
        mu, cov, _, _ = kf.batch_filter(zs, saver=s)
        s.to_array()

        for x, P in zip(mu, cov):
            # covariance of x and y
            cov = np.array([[P[0, 0], P[2, 0]],
                            [P[0, 2], P[2, 2]]])
            mean = (x[0, 0], x[1, 0])
            plot_covariance(mean, cov=cov, fc='g', std=3, alpha=0.5)

        # plot results
        plot_filter(mu[:, 0], mu[:, 1])
        plot_measurements(zs[:, 0], zs[:, 1])
        plt.legend(loc=2)
        # plt.xlim(0, 20)
        plt.show()
        stds = 3
        title = f'First Order Position Residuals({stds}$\sigma$)'
        plot_residuals(xs, s, 0, title=title,y_label='pixels', stds=stds)
        # plot_measurements(zs[:, 0], zs[:, 1])

        plt.show()

    main()
