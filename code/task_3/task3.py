import numpy as np
import cv2
import matplotlib.pyplot as plt
import operator


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
mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix_l.mat(), distMatrix_l.mat(), None, newcameramtx, (w, h), 5)
dst = cv2.remap(img_l, mapx, mapy, cv2.INTER_LINEAR)
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite(r'../../output/task_3/l_distort.png', dst)

# Undistort right image
h, w = img_r.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix_r.mat(), distMatrix_r.mat(), (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix_r.mat(), distMatrix_r.mat(), None, newcameramtx, (w, h), 5)
dst = cv2.remap(img_l, mapx, mapy, cv2.INTER_LINEAR)
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite(r'../../output/task_3/r_distort.png', dst)


#ORB
orb = cv2.ORB_create()
kp_l = orb.detect(gray_l, None)
kp_l, des_l = orb.compute(gray_l, kp_l)
img2_l = cv2.drawKeypoints(gray_l, kp_l, None, color=(0,255,0), flags=0)
plt.imshow(img2_l), plt.show()

keypoint_list = []
for i, keypoint in enumerate(kp_l):
    #print("Keypoint:", i, keypoint)
    keypoint_list.append(keypoint)

# sort by response
cmpfun = operator.attrgetter('response')
keypoint_list.sort(key=cmpfun, reverse=True)

# find minimum
distance = []
radius = []
keypoint_i = 0
for keypoint in keypoint_list:
    # print("Keypoint:", keypoint.response)
    distance.append([])
    if keypoint_i == 0:
        distance[0].append(1)
    for index in range(keypoint_i):
        distance[keypoint_i].append(np.linalg.norm(np.array(keypoint.pt) - np.array(keypoint_list[index].pt)))
    radius.append(min(distance[keypoint_i]))
    # print(keypoint_i, " radius:", radius[keypoint_i])
    keypoint_i = keypoint_i + 1

# sort by suppression radius
keypoint_list = np.c_[keypoint_list, radius]
keypoint_list = sorted(keypoint_list, key=lambda x:x[1], reverse=True)

# get top n = 50
keypoint_list = keypoint_list[0:50]
keypoint_list = np.delete(keypoint_list, 1, axis=1).transpose()[0]

img2_l = cv2.drawKeypoints(gray_l, keypoint_list, None, color=(0,255,0), flags=0)
plt.imshow(img2_l), plt.show()
#print(keypoint_list)
#print(kp_l)
