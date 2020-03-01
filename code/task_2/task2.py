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


fs_l = cv2.FileStorage("../../parameters/left_camera_intrinsics.xml", cv2.FILE_STORAGE_READ)
leftNode = fs_l.getNode("left_camera_intrinsic")

fs_r = cv2.FileStorage("../../parameters/right_camera_intrinsics.xml", cv2.FILE_STORAGE_READ)
rightNode = fs_r("right_camera_intrinsic")


gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)

# Adding 3D point
objpoints.append(objp)

# Adding 2D point
twoDPoint_l = cv2.cornerSubPix(gray_l, corner_l, (11, 11), (-1, -1), criteria)
imgpoints_l.append(twoDPoint_l)
twoDPoint_r = cv2.cornerSubPix(gray_r, corner_r, (11, 11), (-1, -1), criteria)
imgpoints_r.append(twoDPoint_r)

R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, 
