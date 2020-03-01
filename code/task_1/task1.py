# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:47:37 2020
@author: Dell
"""

import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = []
images.append(glob.glob(r'../../images/task_1/left*.png'))
images.append(glob.glob(r'../../images/task_1/right*.png'))

for lr in [0, 1]:
    order = 0
    if lr == 0:
        lr_letter = r'left'
    else:
        lr_letter = r'right'
    for fname in images[lr]:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)
            cv2.imwrite(r'../../output/task_1/' + lr_letter + r'_' + r'%d' % order + r'_calibration.png', img)
        order = order + 1

    order = 0

    # cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    '''
    print ("ret:",ret)
    print ("mtx:\n",mtx)
    print ("dist:\n",dist)
    print ("rvecs:\n",rvecs)
    print ("tvecs:\n",tvecs)
    '''

    # print (np.dtype(mtx))
    # print (np.dtype(dist))

    fs = cv2.FileStorage("../../parameters/" + lr_letter + "_camera_intrinsics.xml", cv2.FILE_STORAGE_WRITE)
    fs.write('camera_intrinsic', mtx)
    fs.write('distort_coefficients', dist)

    for fname in images[0]:
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # print("roi:", roi)

        # undistort
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv2.imwrite(r'../../output/task_1/' + lr_letter + r'_' + r'%d' % order + r'_distort.png', dst)
        order = order + 1

    '''# undistort
    mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
    dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png',dst)'''
