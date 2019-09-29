from src.vision import Vision, MAIN_WINDOW_NAME, DEBUG
from src.object_detection import ball_tracker, ball_tracker_cascade, marker_detect
import cv2 as cv
from src.yaml_parser import Parser
import os


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

        # ball with color
        x_circle, y_circle, radius = ball_tracker(vision.frame, vision.debug, ball_cascade)
        # ball with cascade
        ball_t_cascade = ball_tracker_cascade(vision.frame, ball_cascade)
        # arrow detect
        marker_detect(vision.frame, forward, left, right)

        # draw color balls
        if x_circle > -1 and y_circle > -1 and radius > -1:
            cv.circle(vision.frame, (int(x_circle), int(y_circle)), int(radius), (0,255,255), 2)

        # draw cascade balls
        if ball_t_cascade is not None:
            for (x, y, w, h) in ball_t_cascade:
                cv.rectangle(vision.frame, (x, y), (x+w, y+h), (255,0,0), 2)
        
        cv.imshow(MAIN_WINDOW_NAME, vision.frame)
        cv.imshow(DEBUG, vision.debug)
        # cv.imshow('hsv', vision.hsv)

        if cv.waitKey(1) == ord('q'):
            break

    vision.cap.release()
    cv.destroyAllWindows()