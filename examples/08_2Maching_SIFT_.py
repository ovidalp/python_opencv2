import cv2
from matplotlib import pyplot as plt
import copy
import numpy as np

img1 = cv2.imread('figs/box.png',cv2.IMREAD_GRAYSCALE)          # queryImage
img2 = cv2.imread('figs/box_in_scene.png',cv2.IMREAD_GRAYSCALE) # trainImage
plt.imshow(img1),plt.show()

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
out=copy.copy(img1)

# find the keypoints and descriptors with SIFT

(kp1, des1) = sift.detectAndCompute(img1, None)
(kp2, des2) = sift.detectAndCompute(img2, None)
print("# kps: {}, descriptors: {}".format(len(kp1), des1.shape))
print("# kps: {}, descriptors: {}".format(len(kp2), des2.shape))

# create BFMatcher object
bf = cv2.BFMatcher()

#Match descriptors.
matches = bf.knnMatch(des1,des2, k=2)


# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.65*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,outImg=out,flags=2)

plt.imshow(img3),plt.show()