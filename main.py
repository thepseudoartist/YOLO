import os; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import colorsys

import cv2

import numpy as np

from PIL import Image, ImageFont, ImageDraw

import tensorflow.python.keras.backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import Input

from yolo.model import YOLOEval, YOLOBody, TinyYOLO
from yolo.utils import preprocess

class YOLO(object):
    _def = {
        'model_path': 'model_data/yolo.h5',
        'anchors_path': 'model_data/yolo_anchors.txt',
        'classes_path': 'model_data/coco_classes.txt',
        'score': .5,
        'iou': .3,
        'model_image_size': (320, 320),
        'text_size': 3,
        'gpu_num': 1
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._def: return cls._def[n]
        else: return "Unrecognized attribute name '{}'.".format(n)
    
    def __init__(self, **kwargs):
        self.__dict__.update(self._def)
        self.__dict__.update(kwargs)
        self.class_names = self._get_classes()
        self.anchors = self._get_anchors()
        self.session = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_classes(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path, 'r') as f: class_names = f.readlines(); f.close()

        return [c.strip() for c in class_names]
        
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path, 'r') as f: anchors = f.readline(), f.close()
        
        return np.array([float(x) for x in anchors[0].split(',')]).reshape(-1, 2)
    
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        if_tiny = num_anchors == 6

        try: self.yolo_model = load_model(model_path, compile=False)
        except: 
            self.yolo_model = TinyYOLO(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) if if_tiny else \
                YOLOBody(Input(None, None, 3), num_anchors // 3, num_classes)
            
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
            'MODEL AND GIVEN ANCHOR AND CLASS SIZE MISMATCH'
        
        print('{} model, anchors, and classes loaded.'.format(model_path))

        #Generate colors 
        hsv_tuple = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuple))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        np.random.shuffle(self.colors)

        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = YOLOEval(self.yolo_model.output, self.anchors, len(self.class_names), self.input_image_shape,
                                 score_threshold=self.score, iou_threshold=self.iou)
        
        return boxes, scores, classes
    
    def _detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'should be a multiple of 32'
            assert self.model_image_size[1] % 32 == 0, 'should be a multiple of 32'
            
            boxed_image = preprocess(np.copy(image), tuple(reversed(self.model_image_size)))
            image_data = boxed_image
            
        out_boxes, out_scores, out_classes = self.session.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],
                K.learning_phase(): 0
            }
        )

        thickness = (image.shape[0] + image.shape[1]) // 600
        font_scale = 1
        objects_list = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            scores = '{:.2f}'.format(score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + .5).astype('int32'))
            left = max(0, np.floor(left + .5).astype('int32'))
            bottom = max(0, np.floor(bottom + .5).astype('int32'))
            right = max(0, np.floor(right + .5).astype('int32'))

            mid_h, mid_v = (bottom - top) / 2 + top, (right - left) / 2 + left

            cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)

            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, thickness / self.text_size, 1)

            cv2.rectangle(image, (left, top), (left + text_w, top - text_h - baseline), self.colors[c], thickness=cv2.FILLED)
            cv2.putText(image, label, (left, top - 2), cv2.FONT_HERSHEY_COMPLEX, thickness / self.text_size, (0, 0, 0), 1)

            objects_list.append([top, left, bottom, right, mid_v, mid_h, label, scores])
        
        return image, objects_list
    
    def detect(self, image):
        return self._detect_image(image)

    def close_session(self):
        self.session.close()