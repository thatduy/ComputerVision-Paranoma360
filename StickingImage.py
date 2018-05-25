import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
from numpy import random

def detectAndDescribe(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # check to see if we are using OpenCV 3.X
    if imutils.is_cv3():
        # detect and extract features from the image
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)
    # otherwise, we are using OpenCV 2.4.X
    else:
        # detect keypoints in the image
        detector = cv2.FeatureDetector_create("SIFT")
        kps = detector.detect(gray)
        # extract features from the image
        extractor = cv2.DescriptorExtractor_create("SIFT")
        (kps, features) = extractor.compute(gray, kps)
    return (kps, features)
def sticker(imgs):
    MIN_MATCH_COUNT = 10
    # Initiate SIFT detector opencv 3.x
    (imgLeft,imgRight) = imgs
    # find the keypoints and descriptors with SIFT,call function
    kpRight, desRight = detectAndDescribe(imgRight)
    kpLeft, desLeft = detectAndDescribe(imgLeft)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desRight,desLeft,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kpRight[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpLeft[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        (H, mark) = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        # BIến đổi ảnh bên phải thành cùng không gian với imgLeft qua ma trận H. width = imgRight.width + imgLeft.width
        result = cv2.warpPerspective(imgRight, H, (imgRight.shape[1] + imgLeft.shape[1], imgRight.shape[0]))
        #Ghép imgLeft vào bên trái ảnh result
        result[0:imgLeft.shape[0], 0:imgLeft.shape[1]] = imgLeft
        return result
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
        return None
