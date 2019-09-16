import cv2 as cv
import numpy as np
import math
from yaml_parser import Parser

MAIN_WINDOW_NAME = 'camera'
DEBUG = 'debug'

class Vision:
    def __init__(self, config):
        self.parser = Parser()
        self.min_size = config['min_size']
        self.max_size = config['max_size']

        self.cap = cv.VideoCapture(1)

        col_lb = config['lower_bound']
        col_ub = config['upper_bound']
        self.hsv_vals = {'HL': col_lb[0], 'HU': col_ub[0], 'SL': col_lb[1], 'SU': col_ub[1], 'VL': col_lb[2], 'VU': col_ub[2]}
        self.click_reg = None
        self.debug = None

        cv.namedWindow(MAIN_WINDOW_NAME)
        cv.namedWindow(DEBUG)

        cv.createTrackbar('H (lower)', DEBUG, 0, 255, lambda val: self.update_from_trackbars(val, 'HL'))
        cv.createTrackbar('H (upper)', DEBUG, 0, 255, lambda val: self.update_from_trackbars(val, 'HU'))
        cv.createTrackbar('S (lower)', DEBUG, 0, 255, lambda val: self.update_from_trackbars(val, 'SL'))
        cv.createTrackbar('S (upper)', DEBUG, 0, 255, lambda val: self.update_from_trackbars(val, 'SU'))
        cv.createTrackbar('V (lower)', DEBUG, 0, 255, lambda val: self.update_from_trackbars(val, 'VL'))
        cv.createTrackbar('V (upper)', DEBUG, 0, 255, lambda val: self.update_from_trackbars(val, 'VU'))

        cv.setTrackbarPos('H (lower)', DEBUG, col_lb[0])
        cv.setTrackbarPos('H (upper)', DEBUG, col_ub[0])

        cv.setTrackbarPos('S (lower)', DEBUG, col_lb[1])
        cv.setTrackbarPos('S (upper)', DEBUG, col_ub[1])

        cv.setTrackbarPos('V (lower)', DEBUG, col_lb[2])
        cv.setTrackbarPos('V (upper)', DEBUG, col_ub[2])

        cv.setMouseCallback(MAIN_WINDOW_NAME, self.mouse_cb)


    def update_from_trackbars(self, val, var):
        self.hsv_vals[var] = val

        bound = None
        index = -1
        if var[0] == 'H':
            index = 0
        elif var[0] == 'S':
            index = 1
        elif var[0] == 'V':
            index = 2

        if var[1] == 'U':
            bound = 'upper_bound'
        elif var[1] == 'L':
            bound = 'lower_bound'

        self.parser.write_values(bound, index, val)

        

    def create_hsv(self):
        blurred = cv.GaussianBlur(self.frame, (11,11),0)
        self.hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

        lower_bound = (self.hsv_vals['HL'], self.hsv_vals['SL'], self.hsv_vals['VL'])
        upper_bound = (self.hsv_vals['HU'], self.hsv_vals['SU'], self.hsv_vals['VU'])

        self.debug = cv.inRange(self.hsv, lower_bound, upper_bound) # get binary
        self.debug = cv.erode(self.debug, None, iterations=2)
        self.debug = cv.dilate(self.debug, None, iterations=2)


    def read_camera(self):
        _, self.frame = self.cap.read()
        if not self.cap.isOpened():
            print('Cannot open camera')
            exit()


    def mouse_cb(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.click_reg = [(x,y)]
        elif event == cv.EVENT_LBUTTONUP:
            self.click_reg.append((x,y))
            mean = cv.mean(self.hsv[self.click_reg[0][1]:y, self.click_reg[0][0]:x])
            
            h_mean = int(math.floor(mean[0]))
            s_mean = int(math.floor(mean[1]))
            v_mean = int(math.floor(mean[2]))

            init_bound = 20

            cv.setTrackbarPos('H (lower)', DEBUG, h_mean - init_bound)
            cv.setTrackbarPos('H (upper)', DEBUG, h_mean + init_bound)

            cv.setTrackbarPos('S (lower)', DEBUG, s_mean - init_bound)
            cv.setTrackbarPos('S (upper)', DEBUG, s_mean + init_bound)

            cv.setTrackbarPos('V (lower)', DEBUG, v_mean - init_bound)
            cv.setTrackbarPos('V (upper)', DEBUG, v_mean + init_bound)
