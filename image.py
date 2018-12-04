import numpy as np
import sys
import random
import cv2
import imutils
from scipy import misc

#------------------------------------------------------------------------------
# Finds object based on its pre-defined color
#------------------------------------------------------------------------------
def find_object (img):
    #img = cv2.imread(frame)
    ## convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## mask of green 
    mask = cv2.inRange(hsv, (110, 150, 50), (120, 255,255))

    ## slice the green
    imask = mask>0
    color = np.zeros_like(img, np.uint8)
    color[imask] = img[imask]
    ##show
    #cv2.imshow("object", color)
    ## save 
    #cv2.imwrite("expectedObject.png", color)
    ## return
    cX, cY, boundRect = find_object_centroid (color)

    return cX, cY, boundRect

#------------------------------------------------------------------------------
# Find objects centroid so robot can follow
#------------------------------------------------------------------------------

def find_object_centroid (img):

    # convert it to grayscale, blur it slightly, and threshold it
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 5, 255, cv2.THRESH_BINARY)[1]
    # find contours in the thresholded img
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    # loop over the contours
    contours_poly = [None]*len(cnts)
    boundRect = [None]*len(cnts)
    difX = np.empty(len(cnts))
    i = 0
    contours = False
    cX = None
    cY = None
    ret = None
    #img = np.zeros_like(img, np.uint8)
    for c in cnts:
         
        # draw the contour and center of the shape on the img
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        difX[i] = int(boundRect[i][2])
        contours = True
        i=i+1
        

    if(contours):
        maxX = np.argmax(difX)
        try:
            # compute the center of the contour
            M = cv2.moments(cnts[maxX])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.rectangle(img, (int(boundRect[maxX][0]), int(boundRect[maxX][1])), (int(boundRect[maxX][0]+boundRect[maxX][2]), int(boundRect[maxX][1]+boundRect[maxX][3])), (255, 255, 255), 2)
            cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(img, "Follow", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # show the img
            #cv2.imshow("img", img)
            #cv2.imwrite("centroid.png", img)

            ret = boundRect[maxX]

        except:
             print ("not able to calculate centroid")

    return cX, cY, ret
