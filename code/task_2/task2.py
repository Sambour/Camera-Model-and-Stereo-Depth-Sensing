# -*- coding: utf-8 -*-

import numpy as np
import cv2

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints_l = []  # 2d points in image plane for left camera
imgpoints_r = []  # 2d points in image plane for right camera

img_l = cv2.imread('../../images/task_2/left_0.png')
img_r = cv2.imread('../../images/task_2/right_0.png')

h, w = img_l.shape[:2]

fs_l = cv2.FileStorage("../../parameters/left_camera_intrinsics.xml", cv2.FILE_STORAGE_READ)
cameraMatrix_l = fs_l.getNode("camera_intrinsic")
#print(cameraMatrix_l.mat())
distMatrix_l = fs_l.getNode("distort_coefficients")

fs_r = cv2.FileStorage("../../parameters/right_camera_intrinsics.xml", cv2.FILE_STORAGE_READ)
cameraMatrix_r = fs_r.getNode("camera_intrinsic")
distMatrix_r = fs_r.getNode("distort_coefficients")


gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)

# Adding 3D point
objpoints.append(objp)

# Adding 2D point
twoDPoint_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
imgpoints_l.append(twoDPoint_l)
twoDPoint_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
imgpoints_r.append(twoDPoint_r)


retval, cm1, dc1, cm2, dc2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, cameraMatrix_l.mat(), distMatrix_l.mat(), cameraMatrix_r.mat(), distMatrix_r.mat(), (w, h), criteria = criteria, flags=cv2.CALIB_FIX_INTRINSIC)

# get undistorted points
undist_l = cv2.undistortPoints(twoDPoint_l, cameraMatrix_l.mat(), distMatrix_l.mat())
undist_r = cv2.undistortPoints(twoDPoint_r, cameraMatrix_r.mat(), distMatrix_r.mat())
# cv2.imshow('img',dst)

# print("R:", R)
# print("T:", T)

# calculate two transform matrix, which is [I|0] and [R|t]
I = np.eye(3)
projMatrix_l = np.c_[I, np.zeros(3)]
# print(projMatrix1)
projMatrix_r = np.c_[R, T]
# print(projMatrix2)

# calculate 4D points
points4D = cv2.triangulatePoints(projMatrix_l, projMatrix_r, undist_l, undist_r)
# print(points4D)

# Rectify the stereo camera.
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cm1, dc1, cm2, dc2, (w, h), R, T)

# Check the rectification results

mapx_l, mapy_l = cv2.initUndistortRectifyMap(cm1, dc1, R1, P1, (w, h), 5)
dst_l = cv2.remap(img_l, mapx_l, mapy_l, cv2.INTER_LINEAR)
cv2.imwrite("img_l.png", dst_l)


mapx_r, mapy_r = cv2.initUndistortRectifyMap(cm2, dc2, R2, P2, (w, h), 5)
dst_r = cv2.remap(img_r, mapx_r, mapy_r, cv2.INTER_LINEAR)
cv2.imwrite("img_r.png", dst_r)

fs_sc = cv2.FileStorage("../../parameters/stereo_calibration.xml", cv2.FILE_STORAGE_WRITE)
fs_sc.write('translation_vector', T)
fs_sc.write('rotation_matrix', R)
fs_sc.write('fundamental_matrix', F)
fs_sc.write('essential_matrix', E)

fs_sr = cv2.FileStorage("../../parameters/stereo_rectification.xml", cv2.FILE_STORAGE_WRITE)
fs_sr.write('rectification_rotation_matrix_1', R1)
fs_sr.write('rectification_rotation_matrix_2', R2)
fs_sr.write('rectified_projection_matrix_1', P1)
fs_sr.write('rectified_projection_matrix_2', P2)
fs_sr.write('disparity_depth_matrix', Q)
