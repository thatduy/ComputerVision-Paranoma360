import StickingImage
import imutils
import cv2
imgLeft= cv2.imread('./images/nha_tho_left.jpg') # queryImage
imgRight = cv2.imread('./images/nha_tho_right.jpg') # trainImage
imgLeft = imutils.resize(imgLeft, width=400)
imgRight = imutils.resize(imgRight, width=400)
cv2.imshow("IMAGE LEFT", imgLeft)
cv2.imshow("IMAGE RIGHT", imgRight)
stickedImage = StickingImage.sticker([imgLeft, imgRight])
if stickedImage is not None:
    cv2.imshow("Result", stickedImage)
cv2.waitKey(0)