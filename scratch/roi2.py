"""
Demonstrates a variety of uses for ROI. This class provides a user-adjustable
region of interest marker. It is possible to customize the layout and 
function of the scale/rotate handles in very flexible ways. 
"""

import numpy as np

import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder='row-major')

## Create image to display
arr = np.ones((100, 100), dtype=float)
arr[45:55, 45:55] = 0
arr[25, :] = 5
arr[:, 25] = 5
arr[75, :] = 5
arr[:, 75] = 5
arr[50, :] = 10
arr[:, 50] = 10
arr += np.sin(np.linspace(0, 20, 100)).reshape(1, 100)
arr += np.random.normal(size=(100, 100))

# add an arrow for asymmetry
arr[10, :50] = 10
arr[9:12, 44:48] = 10
arr[8:13, 44:46] = 10

## create GUI
app = pg.mkQApp("ROI Examples")
w = pg.GraphicsLayoutWidget(show=True, size=(1000, 800), border=True)
w.setWindowTitle('pyqtgraph example: ROI Examples')

text = """Data Selection From Image.<br>\n
Drag an ROI or its handles to update the selected image.<br>
Hold CTRL while dragging to snap to pixel boundaries<br>
and 15-degree rotation angles.
"""
w1 = w.addLayout(row=0, col=0)
label1 = w1.addLabel(text, row=0, col=0)
v1a = w1.addViewBox(row=1, col=0, lockAspect=True)
v1b = w1.addViewBox(row=2, col=0, lockAspect=True)
img1a = pg.ImageItem(arr)
v1a.addItem(img1a)
img1b = pg.ImageItem()
v1b.addItem(img1b)
v1a.disableAutoRange('xy')
v1b.disableAutoRange('xy')
v1a.autoRange()
v1b.autoRange()

rois = []
rois.append(pg.RectROI([20, 20], [20, 20], pen=(0, 9), sideScalers=True, removable=True))
rois[-1].addRotateHandle([1, 0], [0.5, 0.5])
# rois.append(pg.LineROI([0, 60], [20, 80], width=5, pen=(1, 9), removable=True))
# rois.append(pg.TriangleROI([80, 75], 20, pen=(5, 9), removable=True))
# # rois.append(pg.MultiRectROI([[20, 90], [50, 60], [60, 90]], width=5, pen=(2, 9)))
# rois.append(pg.EllipseROI([60, 10], [30, 20], pen=(3, 9), removable=True))
# rois.append(pg.CircleROI([80, 50], [20, 20], pen=(4, 9), removable=True))
# rois.append(pg.LineSegmentROI([[110, 50], [20, 20]], pen=(5,9)))
rois.append(pg.PolyLineROI([[80, 60], [90, 30], [60, 40]], pen=(6, 9), closed=True, removable=True))

# p = pg.PolyLineROI([[80, 60], [90, 30], [60, 40]], pen=(6, 9), closed=True)
# a = p.getState()
#
# b = 2

def update(roi):
    img1b.setImage(roi.getArrayRegion(arr, img1a), levels=(0, arr.max()))
    v1b.autoRange()
def remove(roi):
    v1a.removeItem(roi)

for roi in rois:
    roi.sigRegionChanged.connect(update)
    roi.sigRemoveRequested.connect(remove)
    v1a.addItem(roi)

update(rois[-1])

# # Provide a callback to remove the ROI (and its children) when
# # "remove" is selected from the context menu.
# def remove():
#     v4.removeItem(r4)
# #
# #
# # r4.sigRemoveRequested.connect(remove)

if __name__ == '__main__':
    pg.exec()
