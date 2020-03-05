# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
img_r = cv2.imread('../../images/task_2/left_1.png')

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
points4D = [c / points4D[3] for c in points4D]
# print(points4D)

# show projection in 3D
show_points_3D_l = [[0, 0, 0], [1, 1, 3], [1, -1, 3], [-1, -1, 3], [-1, 1, 3]]
show_points_3D_r = []
RTt = np.dot(np.transpose(R), T).transpose()
# print("RTt: ", RTt)
for point in show_points_3D_l:
    point_r = np.dot(np.transpose(R), point) - RTt
    show_points_3D_r.append(point_r)
    # print("Point_r: ", point_r)

square_l = show_points_3D_l
square_l.append(show_points_3D_l[1])
square_r = show_points_3D_r
square_r.append(show_points_3D_r[1])
square_l = np.transpose(square_l[1:6])
square_r = np.transpose(square_r[1:6])

line1_l = np.transpose([show_points_3D_l[1], show_points_3D_l[0], show_points_3D_l[2]])
line2_l = np.transpose([show_points_3D_l[3], show_points_3D_l[0], show_points_3D_l[4]])
line1_r = np.transpose([show_points_3D_r[1], show_points_3D_r[0], show_points_3D_r[2]])
line2_r = np.transpose([show_points_3D_r[3], show_points_3D_r[0], show_points_3D_r[4]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Axes3D.scatter(ax, points4D[0], points4D[1], points4D[2])
Axes3D.plot(ax, square_l[0].flatten(), square_l[1].flatten(), square_l[2].flatten(), 'C1')
Axes3D.plot(ax, line1_l[0].flatten(), line1_l[1].flatten(), line1_l[2].flatten(), 'C1')
Axes3D.plot(ax, line2_l[0].flatten(), line2_l[1].flatten(), line2_l[2].flatten(), 'C1')
Axes3D.plot(ax, square_r[0].flatten(), square_r[1].flatten(), square_r[2].flatten(), 'C2')
Axes3D.plot(ax, line1_r[0].flatten(), line1_r[1].flatten(), line1_r[2].flatten(), 'C2')
Axes3D.plot(ax, line2_r[0].flatten(), line2_r[1].flatten(), line2_r[2].flatten(), 'C2')
plt.savefig("../../output/task_2/Projection_before_rectify.png")

# Rectify the stereo camera.
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cm1, dc1, cm2, dc2, (w, h), R, T)

# Check the rectification results
mapx_l, mapy_l = cv2.initUndistortRectifyMap(cm1, dc1, R1, P1, (w, h), 5)
dst_l = cv2.remap(img_l, mapx_l, mapy_l, cv2.INTER_LINEAR)
x, y, w, h = validPixROI1
dst_l = dst_l[y:y + h, x:x + w]
cv2.imwrite("../../output/task_2/img_l.png", dst_l)

mapx_r, mapy_r = cv2.initUndistortRectifyMap(cm2, dc2, R2, P2, (w, h), 5)
dst_r = cv2.remap(img_r, mapx_r, mapy_r, cv2.INTER_LINEAR)
x, y, w, h = validPixROI2
dst_r = dst_r[y:y + h, x:x + w]
cv2.imwrite("../../output/task_2/img_r.png", dst_r)

# show 3D after rectify
# show projection in 3D
show_original = [[0, 0, 0], [1, 1, 3], [1, -1, 3], [-1, -1, 3], [-1, 1, 3]]
show_points_3D_l = []
show_points_3D_r = []
# print("RTt: ", RTt)
for point in show_original:
    point_l = np.dot(np.transpose(R1), point)
    show_points_3D_l.append(point_l)

RTt = np.dot(np.transpose(R2), T).transpose()
for point in show_original:
    point_r = np.dot(np.transpose(R2), point) - RTt
    show_points_3D_r.append(point_r)
    # print("Point_r: ", point_r)

square_l = show_points_3D_l
square_l.append(show_points_3D_l[1])
square_r = show_points_3D_r
square_r.append(show_points_3D_r[1])
square_l = np.transpose(square_l[1:6])
square_r = np.transpose(square_r[1:6])

line1_l = np.transpose([show_points_3D_l[1], show_points_3D_l[0], show_points_3D_l[2]])
line2_l = np.transpose([show_points_3D_l[3], show_points_3D_l[0], show_points_3D_l[4]])
line1_r = np.transpose([show_points_3D_r[1], show_points_3D_r[0], show_points_3D_r[2]])
line2_r = np.transpose([show_points_3D_r[3], show_points_3D_r[0], show_points_3D_r[4]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Axes3D.scatter(ax, points4D[0], points4D[1], points4D[2])
Axes3D.plot(ax, square_l[0].flatten(), square_l[1].flatten(), square_l[2].flatten(), 'C1')
Axes3D.plot(ax, line1_l[0].flatten(), line1_l[1].flatten(), line1_l[2].flatten(), 'C1')
Axes3D.plot(ax, line2_l[0].flatten(), line2_l[1].flatten(), line2_l[2].flatten(), 'C1')
Axes3D.plot(ax, square_r[0].flatten(), square_r[1].flatten(), square_r[2].flatten(), 'C2')
Axes3D.plot(ax, line1_r[0].flatten(), line1_r[1].flatten(), line1_r[2].flatten(), 'C2')
Axes3D.plot(ax, line2_r[0].flatten(), line2_r[1].flatten(), line2_r[2].flatten(), 'C2')
plt.savefig("../../output/task_2/Projection_after_rectify.png")

# write the parameters
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
