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
            r = requests.get(url, timeout=60)
            
            # save the image to disk
            path = os.path.join(os.getcwd(),'tools/pos') 
            p = os.path.sep.join([path, "{}.jpg".format(
                str(total).zfill(8))])
            
            f = open(p, "wb")
            f.write(r.content)
            f.close()
    
            # update the counter
            print("[INFO] downloaded: {}".format(p))
            total += 1
    
        # handle if any exceptions are thrown during the download process
        except:
            print("[INFO] error downloading {}...skipping".format(p))


def fix_broken():   
    path = os.path.join(os.getcwd(),'tools/pos') 
        
    for imagePath in os.listdir(path):
        if imagePath[:4] != 'ball':
            # initialize if the image should be deleted or not
            delete = False
            pic = os.path.join(path, imagePath)
            # try to load the image
            try:
                image = cv.imread(pic)
        
                # if the image is `None` then we could not properly load it
                # from disk, so delete it
                if image is None:
                    delete = True
                else:
                    img = cv.resize(image, (640,480))
                    cv.imwrite(pic, img)
        
            # if OpenCV cannot load the image then the image is likely
            # corrupt so we should delete it
            except:
                print("Except")
                delete = True
        
            # check to see if the image should be deleted
            if delete:
                print("[INFO] deleting {}".format(imagePath))
                os.remove(pic)


def change_size(target):
    folder = os.getcwd()
    folder = os.path.join(folder, 'tools{0}'.format(target))
    pic_number = 0

    for filename in os.listdir(folder):
        if filename[:3] != 'neg':
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
    pic_number = 180

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
                path = os.path.join(os.getcwd(), 'tools/')
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
                path = os.path.join(os.getcwd(), 'tools/')
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


def fix_info():
    folder = os.getcwd()
    folder = os.path.join(folder, 'tools/info.dat')

    file1 = open(folder,'r')
    file1 = file1.read().split('\n')

    for old_line in file1:
        div = old_line.split()
        if len(div) > 0:
            old_coor = [int(div[2]), int(div[3]), int(div[4]), int(div[5])]
            new_coor = [old_coor[0], old_coor[1], old_coor[2] - old_coor[0], old_coor[3] - old_coor[1]]
            
            line = '{0} {1} {2} {3} {4} {5}\n'.format(div[0], div[1], new_coor[0], new_coor[1], new_coor[2], new_coor[3])

            path = os.path.join(os.getcwd(), 'tools/')
            with open('{}info.lst'.format(path),'a') as f:
                f.write(line)

    # print(file1)

if __name__ == '__main__':
    # parser = Parser()
    # data = parser.data
    # object_to_read = data['object'][0] 
    # vision = Vision(object_to_read['ball'])
    # iterate_circles_position('/pos', vision)


    # negative()
    # create_pos_n_neg()
    # resize_very_small()
    # download_google()
    # fix_broken()

    fix_info()