import numpy as np
import cv2
from matplotlib import pyplot as plt

fig = plt.figure()
a=fig.add_subplot(1,2,1)
img = cv2.imread('figs/butterfly.jpg',0)
png = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create(nonmaxSuppression=1,threshold=100)
# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp,png, color=(255,0,0))

imgplot = plt.imshow(img2)
a.set_title('With nonmaxSuppression')
# Print all default params

#print ("Threshold: ", fast.getInt('threshold'))
#print ("nonmaxSuppression: ", fast.getBool('nonmaxSuppression'))
#print ("neighborhood: ", fast.getInt('type'))
print ("Total Keypoints with nonmaxSuppression: ",len(kp))


#cv2.imwrite('fast_true.png',img2)

#plt.imshow(img2),plt.show()

fast1 = cv2.FastFeatureDetector_create(nonmaxSuppression=0,threshold=100)
# Disable nonmaxSuppression
kp1 = fast1.detect(img,None)

print ("Total Keypoints without nonmaxSuppression: ", len(kp1))
img3 = cv2.drawKeypoints(img, kp1,png, color=(255,0,0))
a=fig.add_subplot(1,2,2)
mgplot = plt.imshow(img3)
a.set_title('Without nonmaxSuppression')
plt.show()



#cv2.imwrite('fast_false.png',img3)