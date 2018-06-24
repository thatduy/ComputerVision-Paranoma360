import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
from numpy import random
import GlobalVar
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
def isRelative(imgs):
    # Initiate SIFT detector opencv 3.x
    (imgLeft,imgRight) = imgs
   
    # find the keypoints and descriptors with SIFT,call function
    kpRight, desRight = detectAndDescribe(imgRight)
    kpLeft, desLeft = detectAndDescribe(imgLeft)
    imgLeft = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2BGRA)
    imgRight = cv2.cvtColor(imgRight, cv2.COLOR_BGR2BGRA)
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
    if len(good) > GlobalVar.MIN_MATCH_COUNT:
        print("inlier", len(good))
        return True
    else:
        print("Not enough matches are found - %d/%d" % (len(good), GlobalVar.MIN_MATCH_COUNT))
        return False
