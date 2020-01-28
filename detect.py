import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import colorsys
import math
import argparse

import cv2

import numpy as np

import tensorflow.python.keras.backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import Input

from yolo.model import YOLOEval, YOLOBody, TinyYOLO
from yolo.utils import preprocess
from main import YOLO

import multiprocessing
from multiprocessing import Pipe

import mss
import time

start_time = time.time()
display_time = 2
fps = 0
sct = mss.mss()
yolo = YOLO()

monitor = {
    'top': 0,
    'left': 0,
    'width': 1080,
    'height': 1080
}

def grab_mss_screen(p_input):
    while True:
        img = np.array(sct.grab(monitor))
        p_input.send(img)

def show_mss_screen(p_output):
    global fps, start_time, yolo

    while True:
        img = p_output.recv()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        r_image, object_list = yolo.detect(img)
        r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)

        fps += 1
        
        TIME = time.time() - start_time

        if TIME >= display_time:
            fps_text = 'FPS: ' + str(math.ceil(fps / TIME))
            fps = 0
            start_time = time.time()

        cv2.putText(r_image, text=fps_text, org=(3, 15), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.50,
                    color=(255, 255, 255), thickness=1)

        cv2.namedWindow("REALTIME OUT", cv2.WINDOW_NORMAL)
        cv2.imshow("REALTIME OUT", r_image)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    yolo.close_session()
    exit()

def _detect_realtime():
    p_output, p_input = Pipe()

    p1 = multiprocessing.Process(target=grab_mss_screen, args=(p_input, ))
    p2 = multiprocessing.Process(target=show_mss_screen, args=(p_output, ))

    p1.run()
    p2.run()

def _detect_webcam():
    global fps, start_time, display_time, yolo
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened(): raise IOError("Cannot start web cam.")

    while True: 
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=1., fy=1., interpolation=cv2.INTER_AREA)

        r_image, objects_list = yolo.detect(frame)
                
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        fps += 1
        TIME = time.time() - start_time
        
        import math

        if TIME >= display_time:
            fps_text = 'FPS: ' + str(math.ceil(fps / TIME))
            fps = 0
            start_time = time.time()
        
        cv2.putText(r_image, text=fps_text, org=(3, 15), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.50,
                    color=(255, 255, 255), thickness=1)

        cv2.namedWindow("WEBCAM OUT", cv2.WINDOW_NORMAL)
        cv2.imshow("WEBCAM OUT", r_image)

    cap.release()
    cv2.destroyAllWindows()
    yolo.close_session()

def _detect_im_and_save(path):
    global yolo

    image = cv2.imread(os.path.expanduser(path))
    image, object_list = yolo.detect(image)

    cv2.imwrite('pred.jpg', image)

    yolo.close_session()

if __name__ == "__main__":
    #_detect_realtime()
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        '--mode', type=int,
        help='realtime(0), webcam(1), image(2)'
    )

    parser.add_argument(
        '--path', type=str,
        required=False,
        help='path of image if working in image mode'
    )

    FLAGS = parser.parse_args()

    if FLAGS.mode == 0: _detect_realtime()
    elif FLAGS.mode == 1: _detect_webcam()
    elif FLAGS.mode == 2: _detect_im_and_save(FLAGS.path)
