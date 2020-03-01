# -*- coding: utf-8 -*-

import numpy as np
import cv2

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

img_l = cv2.imread('../../images/task_2/left_0.png')
img_r = cv2.imread('../../images/task_2/right_0.png')

gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)

