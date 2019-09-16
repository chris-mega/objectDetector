import cv2 as cv
import numpy as np

def ball_tracker(frame, debug):
    _, thresh = cv.threshold(debug.copy(), 127, 255, 0)
    _, contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    center = None


    if len(contours) > 0:
        
        x, y, radius, max_contour = -1, -1, -1, -1
        poly = None
        for i, c in enumerate(contours):
            if max_contour < cv.contourArea(contours[i]):
                poly = cv.approxPolyDP(contours[i], 3, True)
                ((x,y), radius) = cv.minEnclosingCircle(poly)
                max_contour = cv.contourArea(contours[i])

        M = cv.moments(poly)

        c_size = M['m00']

        if c_size > 0:
            center = (int(M['m10'] / c_size), int(M['m01'] / c_size))

            if radius > 5:
                cv.circle(frame, (int(x), int(y)), int(radius), (0,255,255), 2)
                cv.circle(frame, center, 5, (0,0,255), 1)