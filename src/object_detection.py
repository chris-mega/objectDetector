import cv2 as cv
import numpy as np
import os

def ball_tracker_cascade(frame, cascade):
    inp = frame.copy()

    gray = cv.cvtColor(inp, cv.COLOR_BGR2GRAY)


    if cascade is not None:
        balls = cascade.detectMultiScale(gray)

        return balls

    return None
        


def ball_tracker(frame, debug, cascade=None):
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

            if radius <= 5:
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


def marker_detect(frame, forward, left, right):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # edges = cv.Canny(gray, 50, 120)
    # min_line_length = 100
    # max_line_gap = 50
    # lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, min_line_length, max_line_gap)

    # inp = frame.copy()
    # if lines is not None:
    #     for x1, y1, x2, y2 in lines[0]:
    #         cv.line(inp, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # cv.imshow('edges', edges)
    # cv.imshow('lines', inp)

    gray = cv.blur(gray, (3,3))
    _, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)


    edged = cv.Canny(gray, 30, 200)

    # cv.imshow('canny',edged)

    img, contours, hier = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


    posible_arrow = None
    area = -1

    for cnt in contours:
        epsilon = 0.01 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            this_area = cv.contourArea(cnt)
            if area < this_area:
                area = this_area
                posible_arrow = approx.copy()

    if posible_arrow is not None and area > 1000:
        pts1 = np.zeros(shape=(4,2))

        for i, p in enumerate(posible_arrow):
            pts1[i] = p[0]

        pts1.view('i8,i8').sort(order=['f1'], axis=0)

        change_to = np.zeros(shape=(4,2))

        if pts1[0][0] < pts1[1][0]:
            change_to[0] = [0,0]
            change_to[1] = [400,0]
        else:
            change_to[0] = [400,0]
            change_to[1] = [0,0]
        
        if pts1[2][0] < pts1[3][0]:
            change_to[2] = [0,400]
            change_to[3] = [400,400]
        else:
            change_to[2] = [400,400]
            change_to[3] = [0,400]


        M = cv.getPerspectiveTransform(np.float32(pts1), np.float32(change_to))

        dst = cv.warpPerspective(frame, M, (400, 400))

        cv.imshow('dst', dst)        

        orb = cv.ORB_create()

        kp1, des1 = orb.detectAndCompute(forward, None)
        kp2, des2 = orb.detectAndCompute(dst, None)

        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des1, des2)

        if len(matches) > 4:
            print('forward')
        else:
            print('nop')

        cv.drawContours(frame, [posible_arrow], 0, (0,255,0), 2)


