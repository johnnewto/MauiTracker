import numpy as np
import cv2

cols, rows = 1920, 2176

pixels = np.fromfile("rgp12.raw", np.uint8)

""" CONVERT THE BYTE STREAM, EVERY PIXEL HAS 12 BIT, SO BYTE HAS TO BE SPLITTED AND PUTTED IN A UINT16 VARIABLE"""
data = pixels
data1 = data.astype(np.uint16)

result = np.zeros(data.size*2//3, np.uint16)

# 12 bits packing: ######## ######## ########
#                  | 8bits| | 4 | 4  |  8   |
#                  |  lsb | |msb|lsb |  msb |
#                  <-----------><----------->
#                     12 bits       12 bits

result[0::2] = ((data1[1::3] & 15) << 8) | data1[0::3]
result[1::2] = (data1[1::3] >> 4) | (data1[2::3] << 4)
bayer_im = np.reshape(result, (rows, cols))

bgr = cv2.cvtColor(bayer_im, cv2.COLOR_BayerBG2BGR)
cv2.imshow('bgr', bgr*16)

# "White balance":
bgr[:, :, 0] = np.minimum(bgr[:, :, 0].astype(np.float32)*1.8, 4095).astype(np.uint16)
bgr[:, :, 2] = np.minimum(bgr[:, :, 2].astype(np.float32)*1.67, 4095).astype(np.uint16)

cv2.imshow('bayer_im', bayer_im*16)
cv2.imshow('bgr WB', bgr*16)
cv2.imshow('bgr WB_2', bgr)


cv2.imwrite('processed.tif',bgr*16 )

cv2.waitKey()
cv2.destroyAllWindows()