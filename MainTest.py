import StickingImage
import imutils
import GlobalVar
import FindCenter
import cv2
import os
from ImageInfo import ImageInfoClass
import threading
import time
import numpy as np
from matplotlib import pyplot as plt
def findInfoImage(file1):
    global list_images
    tempImage = ImageInfoClass(0, "", [])
    tempImage.name = file1
    for file2 in os.listdir('./test'):
        if file1 == file2:
            continue
        image1 = imutils.resize(cv2.imread('./test/%s'%file1), width = 400)
        image2 = imutils.resize(cv2.imread('./test/%s'%file2), width = 400)
        if FindCenter.isRelative([image1, image2]) == True:
            tempImage.count  = tempImage.count + 1
            tempImage.relatitons.append(file2)
    list_images.append(tempImage)

list_images = []
threads = []

for file1 in os.listdir('./test'):
    t1 = threading.Thread(target=findInfoImage, args=(file1,))
    t1.start()
    #time.sleep(1)
    threads.append(t1)
for t in threads:
    t.join()
tempImage = ImageInfoClass(0, "", [])
firstImage  = ImageInfoClass(0, "", [])
imgCenterClass = ImageInfoClass(0, "", [])
sortedImages = []
onlyHorizontal = True
for im in list_images:
    if im.count > tempImage.count:
        tempImage = im
        print(im.name, im.count, im.relatitons)
    if im.count > 2:
            onlyHorizontal = False
    if im.count == 1:
            firstImage = im
if onlyHorizontal and len(list_images) > 3:
    sortedImages.append(firstImage)
    tempImg = firstImage
    while len(list_images) > 1:
        for img in list_images:
            if img.name in tempImg.relatitons:
                sortedImages.append(img)
                list_images.remove(tempImg)
                tempImg = img
    imgCenterClass = sortedImages[int(np.ceil(len(sortedImages)/2)) - 1]
    imgCenter = imutils.resize(cv2.imread('./test/%s'%imgCenterClass.name), width = 400)
    list_images = sortedImages
else:
    list_images.remove(tempImage)
    imgCenter = imutils.resize(cv2.imread('./test/%s'%tempImage.name), width = 400)
index = 0
while len(list_images) > 0:
    if index == len(list_images):
        print("-----------------------TRY AGAIN--------------------")
        GlobalVar.MIN_MATCH_COUNT-=5
        index = 0
        if GlobalVar.MIN_MATCH_COUNT < 5:
            break
        continue
    image2 = imutils.resize(cv2.imread('./test/%s'%list_images[index].name), width = 400)
    temp = StickingImage.sticker([imgCenter, image2])
    if temp is not None:
        imgCenter = imutils.resize(temp, width = 600)
        list_images.remove(list_images[index])
        GlobalVar.MIN_MATCH_COUNT=50
        index = 0
    else:
        index+=1
cv2.imwrite("panorama.png", imgCenter)
# cv2.imshow("result", imgCenter)
# cv2.waitKey(0)
plt.imshow(imgCenter)
plt.show()