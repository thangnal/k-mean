import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt

sausage= cv2.imread('/Users/thangna/Downloads/comm/categorize/IMG_0906 copy.JPG',0)
chip = cv2.imread('/Users/thangna/Downloads/comm/categorize/IMG_1043.JPG',0)
meat = cv2.imread('/Users/thangna/Downloads/comm/categorize/IMG_0739.JPG',0)

test_img = cv2.imread('/Users/thangna/Downloads/comm/403/20170905_104600.JPG',0)

def predict(train_img, test_img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(test_img, None)
    kp2, des2 = sift.detectAndCompute(train_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck= True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x:x.distance)

    # print len(matches)
    return len(matches)

# print predict(sausage, test_img)
# print predict(chip, test_img)

# img3 = cv2.drawMatches(sausage, kp1, test_img, kp2, matches,None, flags = 4)

# plt.imshow(img3), plt.show()

i = 0
count = 0
for n in glob.glob("/Users/thangna/Downloads/comm/402/*.JPG"):

    img = cv2.imread(n, 0)

    compare = [predict(chip, img), predict(sausage, img), predict(meat, img)]
    # compare_mse = [mse(chip, img), mse(meat, img), mse(sausage, img)]
    # print compare
    if compare.index(max(compare)) == 0:
        print "It's chip"
    elif compare.index(max(compare)) == 1:
        print "It's sausage"
    elif compare.index(max(compare)) == 2:
        count += 1
        print "It's meat"
    i += 1

print i
print count


