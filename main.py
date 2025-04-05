from src.vision import Vision, MAIN_WINDOW_NAME, DEBUG
from src.object_detection import ball_tracker, ball_tracker_cascade, marker_detect
import cv2 as cv
from src.yaml_parser import Parser
import os
import math


if __name__ == '__main__':
    with_cascade = True
    parser = Parser()
    data = parser.data
    object_to_read = data['object'][0] 
    vision = Vision(object_to_read['ball'])
    ball_cascade = None

    folder = os.getcwd()
    folder = os.path.join(folder, 'src')
    forward = cv.imread('{}/forward.png'.format(folder), 0)
    left = cv.imread('{}/left.png'.format(folder), 0)
    right = cv.imread('{}/right.png'.format(folder), 0)

    if with_cascade:
        path = os.getcwd()
        path = os.path.join(path, 'tools/data/cascade.xml')
        ball_cascade = cv.CascadeClassifier(path)
    
    while True:
        vision.read_camera()
        vision.create_hsv()

        area1, area2 = -1, -1

        # ball with color
        x_circle, y_circle, radius = ball_tracker(vision.frame, vision.debug, min_size=vision.min_size)

        if radius > -1:
            area1 = math.pi*(radius**2)
            print('[BALL color] x: {0}\ty: {1}\tarea: {2}'.format(x_circle, y_circle, area1))

        
        # ball with cascade
        ball_c_x, ball_c_y, ball_c_w, ball_c_h = ball_tracker_cascade(vision.frame, ball_cascade, min_size=vision.min_size)
        
        if ball_c_w > -1 and ball_c_w > -1:
            area2 = ball_c_w * ball_c_h
            x = ball_c_x + ball_c_w/2
            y = ball_c_y + ball_c_h/2
            print('[BALL cascade] x: {0}\ty: {1}\tarea: {2}'.format(x, y, area2))        


        # arrow detect
        arrow = marker_detect(vision.frame, forward, left, right)
        # print('[ARROW] {}'.format(arrow))

        # combination cascade and color
        if radius > -1 and ball_c_w > -1 and ball_c_w > -1 and abs((ball_c_x+ ball_c_w/2) - x_circle) < 20:
            area = (area1 + area2)/2.
            x = int((ball_c_x + ball_c_w/2 + x_circle)/2)
            y = int((ball_c_y + ball_c_h/2 + y_circle)/2)

            w = int((area**(0.5)))

            cv.rectangle(vision.frame, (x-w//2, y-w//2), (x + w//2, y + w//2), (0,0,0), 2)

            print('[BALL estimated] x: {0}\ty: {1}\tarea: {2}'.format(x, y, area))

        # draw color balls
        if x_circle > -1 and y_circle > -1 and radius > -1:
            cv.circle(vision.frame, (int(x_circle), int(y_circle)), int(radius), (0,255,255), 2)

        # draw cascade balls
        if ball_c_x > -1 and ball_c_y > -1 and ball_c_w > -1 and ball_c_h > -1:
            cv.rectangle(vision.frame, (ball_c_x, ball_c_y), (ball_c_x + ball_c_w, ball_c_y + ball_c_h), (255,0,0), 2)
        
        cv.imshow(MAIN_WINDOW_NAME, vision.frame)
        cv.imshow(DEBUG, vision.debug)

        if cv.waitKey(1) == ord('q'):
            break

    vision.cap.release()
    cv.destroyAllWindows()
    