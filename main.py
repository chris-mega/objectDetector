from src.vision import Vision, MAIN_WINDOW_NAME, DEBUG
from src.object_detection import ball_tracker
import cv2 as cv
from src.yaml_parser import Parser


if __name__ == '__main__':
    parser = Parser()
    data = parser.data
    object_to_read = data['object'][0] 
    vision = Vision(object_to_read['ball'])
    
    while True:
        vision.read_camera()
        vision.create_hsv()

        ball_tracker(vision.frame, vision.debug)

        cv.imshow(MAIN_WINDOW_NAME, vision.frame)
        cv.imshow(DEBUG, vision.debug)
        # cv.imshow('hsv', vision.hsv)

        if cv.waitKey(1) == ord('q'):
            break

    vision.cap.release()
    cv.destroyAllWindows()