import cv2
from matplotlib import pyplot as plt
import copy

img1 = cv2.imread('figs/box.png',cv2.IMREAD_GRAYSCALE)          # queryImage
img2 = cv2.imread('figs/box_in_scene.png',cv2.IMREAD_GRAYSCALE) # trainImage

# Initiate ORB detector
orb = cv2.ORB_create()
out=copy.copy(img1)

# find the keypoints and descriptors with ORB
# find the keypoints with ORB
kp1 = orb.detect(img1,None)
kp2 = orb.detect(img2,None)
# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img2, kp2)
print("# kps: {}, descriptors: {}".format(len(kp1), des1.shape))
print("# kps: {}, descriptors: {}".format(len(kp2), des2.shape))

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#Match descriptors.
matches = bf.match(des1,des2)


#Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
#print(matches[:10])
# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],outImg=out,flags=2)

plt.imshow(img3),plt.show()