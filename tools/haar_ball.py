#! /usr/bin/env python
import urllib
import numpy as np
import os
import requests
from src.vision import Vision
from src.object_detection import ball_tracker
import cv2 as cv
from src.yaml_parser import Parser

def download_google():
    folder = os.getcwd()
    folder = os.path.join(folder, 'tools/urls.txt')
    rows = open(folder).read().strip().split('\n')
    total = 0

    p = None

    for url in rows:
        try:
            # try to download the image
            r = requests.get(url, auth=('user', 'pass'))
            print(here)
            # save the image to disk
            p = os.path.sep.join([args["output"], "{}.jpg".format(
                str(total).zfill(8))])
            print(p)
            f = open(p, "wb")
            f.write(r.content)
            f.close()
    
            # update the counter
            print("[INFO] downloaded: {}".format(p))
            total += 1
    
        # handle if any exceptions are thrown during the download process
        except:
            print("[INFO] error downloading {}...skipping".format(p))
        break

    # for imagePath in paths.list_images(args["output"]):
    #     # initialize if the image should be deleted or not
    #     delete = False
    
    #     # try to load the image
    #     try:
    #         image = cv2.imread(imagePath)
    
    #         # if the image is `None` then we could not properly load it
    #         # from disk, so delete it
    #         if image is None:
    #             delete = True
    
    #     # if OpenCV cannot load the image then the image is likely
    #     # corrupt so we should delete it
    #     except:
    #         print("Except")
    #         delete = True
    
    #     # check to see if the image should be deleted
    #     if delete:
    #         print("[INFO] deleting {}".format(imagePath))
    #         os.remove(imagePath)

def change_size(target):
    folder = os.getcwd()
    folder = os.path.join(folder, 'tools{0}'.format(target))
    pic_number = 0

    for filename in os.listdir(folder):
        pic = os.path.join(folder, filename)
        img = cv.imread(pic, cv.IMREAD_GRAYSCALE)
        
        resized_img = cv.resize(img, (640,480))
        cv.imwrite('{0}/neg_{1}.jpg'.format(folder, pic_number), resized_img)
        pic_number += 1


def negative():
    change_size('/neg')


def iterate_circles_position(target, vision):
    folder = os.getcwd()
    folder = os.path.join(folder, 'tools{0}'.format(target))
    pic_number = 0

    for filename in os.listdir(folder):
        if filename[:4] != 'ball':
            k = None
            pic = os.path.join(folder, filename)
            img = cv.imread(pic)
            img = cv.resize(img, (640,480))
            
            x, y, size = -1, -1, -1
            while True:
                vision.frame = img.copy()
                vision.create_hsv()
                x, y, size = ball_tracker(vision.frame, vision.debug)
                cv.imshow('camera', vision.frame)
                cv.imshow('debug', vision.debug)
                k = cv.waitKey(1)
                if k == ord('s') or k == ord('n'): # s = save, n = next
                    break
            
            print('ball_{}'.format(pic_number), x, y, size)

            if k == ord('s'):
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                
                cv.imwrite('{0}/ball_{1}.jpg'.format(folder, pic_number), gray)
                
                line = 'pos/ball_{0}.jpg 1 {1} {2} {3} {4}\n'.format(pic_number, int(round(x-size)), int(round(y-size)), int(round(x+size)), int(round(y+size)))
                path = os.path.join(folder, 'tools/')
                with open('{}info.dat'.format(path),'a') as f:
                    f.write(line)

            pic_number += 1


def create_pos_n_neg():
    target = '/neg'
    folder = os.getcwd()
    folder = os.path.join(folder, 'tools{0}'.format(target))

    file_type = 'neg'

    for img in os.listdir(folder):
        if file_type == 'pos':
            line = file_type+'/'+img+' 1 0 0 50 50\n'
            with open('info.dat','a') as f:
                f.write(line)
        elif file_type == 'neg':
            if img[:3] == 'neg':
                line = file_type+'/'+img+'\n'
                path = os.path.join(folder, 'tools/')
                with open('{}bg.txt'.format(path),'a') as f:
                    f.write(line)


def resize_very_small():
    target = '/neg'
    folder = os.getcwd()
    folder = os.path.join(folder, 'tools{0}'.format(target))
    for filename in os.listdir(folder):
        if filename[:3] != 'neg':
            pic = os.path.join(folder, filename)
            img = cv.imread(pic)
            img = cv.resize(img, (640,480))
            cv.imwrite(pic, img)


if __name__ == '__main__':
    # parser = Parser()
    # data = parser.data
    # object_to_read = data['object'][0] 
    # vision = Vision(object_to_read['ball'])
    # iterate_circles_position('/pos', vision)


    # negative()
    # create_pos_n_neg()
    # resize_very_small()
    download_google()