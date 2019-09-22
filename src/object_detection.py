import cv2 as cv
import numpy as np

def ball_tracker(frame, debug):
    inp = frame.copy()
    # with color
    _, thresh = cv.threshold(debug.copy(), 127, 255, 0)
    _, contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    center = None

    x, y, radius = -1, -1, -1

    if len(contours) > 0:
        
        max_contour = -1
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
            else:
                x, y, radius = -1, -1, -1
        else:
            x, y, radius = -1, -1, -1

    # without color, just circle detection
    # gray = cv.cvtColor(inp, cv.COLOR_BGR2GRAY)
    # cv.imshow('gray',gray)
    # edges = cv.Canny(gray, 100, 200)
    # circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.2, 100)
    # if circles is not None:
    #     circles = np.round(circles[0,:]).astype('int')

    #     for (x,y,r) in circles:
    #         cv.rectangle(frame, (x - r, y - r), (x + r, y + r), (0, 255, 0), 2)

    return x, y, radius


