import numpy as np
import cv2
import matplotlib as plt

img_l = cv2.imread('../../images/task_3_and_4/left_0.png')
img_r = cv2.imread('../../images/task_3_and_4/right_0.png')


gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

fs = cv2.FileStorage("../../parameters/stereo_rectification.xml", cv2.FILE_STORAGE_READ)
disparityMatrix = fs.getNode("disparity_depth_matrix")

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=13)  
disparity = stereo.compute(gray_l,gray_r)
depth = cv2.reprojectImageTo3D(disparity, disparityMatrix)
plt.imshow(disparity,'gray')
plt.show()
