from skimage.measure import compare_ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

#load image
img1 = cv2.imread("//Users/thangna/Downloads/comm/401/20171002_105822.JPG",0)
img2 = cv2.imread("//Users/thangna/Downloads/comm/401/20171002_105818.JPG",0)
img3 = cv2.imread("//Users/thangna/Downloads/comm/401/20171002_105816.JPG",0)

img1 = cv2.resize(img1, (600, 900))
img2 = cv2.resize(img2, (600, 900))
img3 = cv2.resize(img3, (600, 900))

# cv2.imshow('Test image',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def mean_squared_error(imgA, imgB):
    mse =  np.sum((imgA.astype("float") - imgB.astype("float")) ** 2)
    mse /= float(imgA.shape[0] * imgA.shape[1])
    return mse

def compare_image(imgA, imgB, title):
    mse = mean_squared_error(imgA,imgB)
    ssim = compare_ssim(imgA,imgB)

    figure = plt.figure(title)
    plt.suptitle("MSE: %2f, SSIM: %2f" % (mse,ssim))

    # show imgA
    figure.add_subplot(1,2,1)
    plt.imshow(imgA, cmap=plt.cm.gray)
    plt.axis("off")

    # show imgB
    figure.add_subplot(1,2,2)
    plt.imshow(imgB, cmap=plt.cm.gray)
    plt.axis("off")

    # show plot
    plt.show()

compare_image(img1,img2,"img1 vs img2")
# compare_image(img2,img3,"img2 vs img3")
# compare_image(img3,img1,"img3 vs img1")