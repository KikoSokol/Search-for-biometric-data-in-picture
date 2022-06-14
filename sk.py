import cv2 as cv2
import sys
import copy
import pandas as pd
import numpy as np

img = cv2.imread('duhovky/001/R/S1001R01.jpg', 0)
img = cv2.GaussianBlur(img, (3, 3), 5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 33, 321,
                              param1=128, param2=322, minRadius=36, maxRadius=154)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    img = cimg

cv2.imshow('detected circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('duhovky/001/R/S1001R08.jpg', 0)
img = cv2.GaussianBlur(img, (5, 5), 5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_blur = cv2.medianBlur(img, 5)
circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 500, param1=80, param2=26, minRadius=10, maxRadius=275)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    img = cimg

cv2.imshow('detected circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()