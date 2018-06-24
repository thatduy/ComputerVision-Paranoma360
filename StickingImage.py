import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
from numpy import random
import GlobalVar

def get_stitched_image(img1, img2, M):

	# Get width and height of input images	
	w1,h1 = img1.shape[:2]
	w2,h2 = img2.shape[:2]

	# Get the canvas dimesions
	img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
	img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)


	# Get relative perspective of second image
	img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

	# Resulting dimensions
	result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)

	# Getting images together
	# Calculate dimensions of match points
	[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel())
	[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel())
	
	# Create output array after affine transformation 
	transform_dist = [-x_min,-y_min]
	transform_array = np.array([[1, 0, transform_dist[0]], 
								[0, 1, transform_dist[1]], 
								[0,0,1]]) 

	# Warp images to get the resulting image
	result_img = cv2.warpPerspective(img2, transform_array.dot(M), 
									(x_max-x_min, y_max-y_min))
	result_img[transform_dist[1]:w1+transform_dist[1], 
				transform_dist[0]:h1+transform_dist[0]] = img1

	# Return the result
	return result_img

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

    # Initiate SIFT detector opencv 3.x
    (imgLeft,imgRight) = imgs
   
    # find the keypoints and descriptors with SIFT,call function
    kpRight, desRight = detectAndDescribe(imgRight)
    kpLeft, desLeft = detectAndDescribe(imgLeft)
    # make image has 4 chanel --> transparent
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
        src_pts = np.float32([ kpRight[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpLeft[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        (H, mark) = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        #(size, offset) = calculate_size(imgLeft.shape, imgRight.shape, H)
        result =  get_stitched_image(imgLeft, imgRight, H)
        #merge_images(imgLeft, imgRight, H, size, offset)
        return result
    else:
        print("Not enough matches are found - %d/%d" % (len(good), GlobalVar.MIN_MATCH_COUNT))
        matchesMask = None
        return None
