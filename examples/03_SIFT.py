import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('figs/france_tour.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img2 = np.float32(gray)
sift = cv2.xfeatures2d.SIFT_create(100)
#sift = cv2.SIFT()
(kps,descs) = sift.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
img2 = cv2.drawKeypoints(gray,kps,img2,255, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#cv2.imshow('test',img2)
#cv2.imwrite('sift_keypoints.jpg',img2)

plt.imshow(img2),plt.show()