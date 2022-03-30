import cv2
 
from display import plotImage
from compareSSIM import compare_ssim
import numpy as np
import imutils

def compareImage(master,test):
    master = cv2.cvtColor(master,cv2.COLOR_BGR2GRAY)
    test = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
    _,matched = compare_ssim(master,test,full=True)
    # plotImage(matched)
    defect,_ =postProcessing(matched,master)
    return defect

def comparePages(masterList,testList):
    for (master,test) in zip(masterList,testList):
        defect = compareImage(master,test)
        plotImage(defect)

def postProcessing(diff,img2):
    diff = (diff * 255).astype("uint8")
    diff = np.uint8(diff)
    thresh = cv2.threshold(diff, 100, 255,
    cv2.THRESH_BINARY_INV)[1]
    plotImage(diff)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    area = 0
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        area +=  cv2.contourArea(c)
            
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 0), 2)
    # displayImage(thresh,'thresh')
    # displayImage(img2,'change')
    return img2,area
