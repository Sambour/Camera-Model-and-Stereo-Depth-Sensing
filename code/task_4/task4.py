import numpy as np
import cv2
import matplotlib.pyplot as plt

img_l = cv2.imread('../../images/task_3_and_4/left_0.png')
img_r = cv2.imread('../../images/task_3_and_4/right_0.png')

gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

fs_l = cv2.FileStorage("../../parameters/left_camera_intrinsics.xml", cv2.FILE_STORAGE_READ)
cameraMatrix_l = fs_l.getNode("camera_intrinsic")
#print(cameraMatrix_l.mat())
distMatrix_l = fs_l.getNode("distort_coefficients")

fs_r = cv2.FileStorage("../../parameters/right_camera_intrinsics.xml", cv2.FILE_STORAGE_READ)
cameraMatrix_r = fs_r.getNode("camera_intrinsic")
distMatrix_r = fs_r.getNode("distort_coefficients")

fs = cv2.FileStorage("../../parameters/stereo_rectification.xml", cv2.FILE_STORAGE_READ)
rotationMatrix_l = fs.getNode("rectification_rotation_matrix_1")
rotationMatrix_r = fs.getNode("rectification_rotation_matrix_2")
projectMatrix_l = fs.getNode("rectified_projection_matrix_1")
projectMatrix_r = fs.getNode("rectified_projection_matrix_2")
disparityMatrix = fs.getNode("disparity_depth_matrix")

# Undistort left image
h, w = img_l.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix_l.mat(), distMatrix_l.mat(), (w, h), 1, (w, h))
mapx_l, mapy_l = cv2.initUndistortRectifyMap(cameraMatrix_l.mat(), distMatrix_l.mat(), rotationMatrix_l.mat(), projectMatrix_l.mat(), (w, h), 5)
dst_l = cv2.remap(img_l, mapx, mapy, cv2.INTER_LINEAR)
x, y, w, h = roi
dst_l = dst_l[y:y + h, x:x + w]
gray_l = cv2.cvtColor(dst_l, cv2.COLOR_BGR2GRAY)

# Undistort right image
h, w = img_r.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix_r.mat(), distMatrix_r.mat(), (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix_r.mat(), distMatrix_r.mat(), rotationMatrix_r.mat(), projectMatrix_r.mat(), (w, h), 5)
dst_r = cv2.remap(img_r, mapx, mapy, cv2.INTER_LINEAR)
x, y, w, h = roi
dst_r = dst_r[y:y + h, x:x + w]
gray_r = cv2.cvtColor(dst_r, cv2.COLOR_BGR2GRAY)
plt.imshow(dst_r)


stereo = cv2.StereoBM_create(numDisparities=16, blockSize=13)  
#disparity = stereo.compute(gray_l, gray_r)
#depth = cv2.reprojectImageTo3D(disparity, disparityMatrix.mat())
#plt.imshow(disparity,'gray')
#plt.imshow(dst_l)
plt.show()
