import numpy as np
import cv2


img_l = cv2.imread('../../images/task_3_and_4/left_0.png')
img_r = cv2.imread('../../images/task_3_and_4/right_0.png')

images = []
images.append(img_l)
images.append(img_r)
gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

fs_l = cv2.FileStorage("../../parameters/left_camera_intrinsics.xml", cv2.FILE_STORAGE_READ)
cameraMatrix_l = fs_l.getNode("camera_intrinsic")
#print(cameraMatrix_l.mat())
distMatrix_l = fs_l.getNode("distort_coefficients")

fs_r = cv2.FileStorage("../../parameters/right_camera_intrinsics.xml", cv2.FILE_STORAGE_READ)
cameraMatrix_r = fs_r.getNode("camera_intrinsic")
distMatrix_r = fs_r.getNode("distort_coefficients")


# Undistort left image
h, w = img_l.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix_l.mat(), distMatrix_l.mat(), (w, h), 1, (w, h))
print(roi)
mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix_l.mat(), distMatrix_l.mat(), None, newcameramtx, (w, h), 5)
dst = cv2.remap(img_l, mapx, mapy, cv2.INTER_LINEAR)
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite(r'../../output/task_3/l_distort.png', dst)
