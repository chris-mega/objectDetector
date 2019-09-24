from src.vision import Vision, MAIN_WINDOW_NAME, DEBUG
from src.object_detection import ball_tracker
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

    if with_cascade:
        path = os.getcwd()
        path = os.path.join(path, 'tools/data/cascade.xml')
        ball_cascade = cv.CascadeClassifier(path)
    
    while True:
        vision.read_camera()
        vision.create_hsv()

        ball_tracker(vision.frame, vision.debug, ball_cascade)

        cv.imshow(MAIN_WINDOW_NAME, vision.frame)
        cv.imshow(DEBUG, vision.debug)
        # cv.imshow('hsv', vision.hsv)

        if cv.waitKey(1) == ord('q'):
            break

    vision.cap.release()
    cv.destroyAllWindows()