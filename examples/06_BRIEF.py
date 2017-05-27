import numpy as np
import sys
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('figs/france_tour.jpg',0)
# Initiate STAR detector
star = cv2.xfeatures2d.StarDetector_create()

# Initiate BRIEF extractor
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# find the keypoints with STAR
kp = star.detect(img,None)

# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)
print("# kps: {}, descriptors: {}".format(len(kp), des.shape))
img2 = cv2.drawKeypoints(img, kp,img, color=(255,0,0))
plt.imshow(img2),plt.show()