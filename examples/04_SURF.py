import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('figs/butterfly.jpg')

#plt.imshow(img),plt.show()

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv2.xfeatures2d.SURF_create(40000)
# Find keypoints and descriptors directly
kps, des = surf.detectAndCompute(img,None)
print (len(kps))
print (surf.descriptorSize())
img2 = cv2.drawKeypoints(img,kps,img,255,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(img2),plt.show()
#surf.hessianThreshold = 50000
# We set it to some 50000. Remember, it is just for representing in picture.
# In actual cases, it is better to have a value 300-500
# surf.hessianThreshold = 50000
# # Again compute keypoints and check its number.
# kp, des = surf.detectAndCompute(img,None)
# print (len(kp))