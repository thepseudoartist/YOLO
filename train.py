import numpy as np

import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Input, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo.model import PreprocessTrueBoxes, YOLOBody, TinyYOLO, YOLOLoss
from yolo.utils import get_rand_data

def _main():
    annotation_path = 'train.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/coco_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = _get_classes(classes_path)
    num_classes = len(class_names)
    anchors = _get_anchors(anchors_path)

    input_shape = (416, 416)

    is_tiny = len(anchors) == 6
    
    if is_tiny: model = create_tiny(input_shape, anchors, num_classes,
                                  freeze_body=2, weights_path='model_data/yolo_tiny.h5')
    else: model = create(input_shape, anchors, num_classes, 
                        freeze_body=2, weights_path='model_data/yolo.h5')

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch: 04d} - loss{loss: .4f} - val_loss{val_loss: .4f}.h5',
                                monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = .1
    
    with open(annotation_path) as f:
        lines = f.readlines()
    
    np.random.seed(1)
    np.random.shuffle(lines)

    np.random.seed(None)

    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # Training Frozen layers first.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })

        batch_size = 32

        print('Train on {} samples, val on {} samples, with batch size {}'.format(
            num_train,
            num_val,
            batch_size
        ))

        model.fit_generator(_data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=_data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes), 
            validation_steps=max(1, num_val // batch_size),
            epochs=50,
            initial_epoch=0,
            callbacks=[logging, checkpoint])
        
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    #Unfreeze and continue training, to fine tune.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        
        #Recompile
        model.compile(optimizer=Adam(lr=1e-4), loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })
        
        model.summary()
        return

        batch_size = 32

        print('Unfreeze all of the layers.')
        print('Train on {} samples, val on {} samples, with batch size {}'.format(
            num_train, num_val, batch_size
        ))

        model.fit_generator(_data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes), 
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=_data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val // batch_size),
            epochs=100,
            initial_epoch=50,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])

        model.save_weights(log_dir + 'trained_weights_final.h5')

def _get_classes(classes_path):
    with open(classes_path) as f:
        _class_names = f.readlines()
    
    _class_names = [c.strip() for c in _class_names]
    return _class_names

def _get_anchors(anchors_path):
    with open(anchors_path) as f:
        _anchors = f.readline()

    _anchors = [float(x) for x in _anchors.split(',')]
    return np.array(_anchors).reshape(-1, 2)


def create(input_shape, 
           anchors, 
           num_classes, 
           load_pretrained=True, 
           freeze_body=2,
           weights_path='model_data/yolo_weights.h5' ):
    
    K.clear_session()
    
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], num_anchors // 3, num_classes + 5))
              for l in range(3)]
    
    model_body = YOLOBody(image_input, num_anchors // 3, num_classes)
    
    print('Create YOLO3 model with {} anchors and {} classes.'.format(
        num_anchors,
        num_classes
    ))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True)
        
        print('Load weights {}'.format(weights_path))

        if freeze_body in [1, 2]:
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            
            for i in range(num): model_body.layers[i].trainable = False

            print('Freeze the first {} layers of total {} layers'.format({
                num,
                len(model_body.layers)
            }))

    model_loss = Lambda(YOLOLoss, output_shape=(1, ), name='yolo_loss', arguments={
        'anchors': anchors,
        'num_classes': num_classes,
        'ignore_threshold': .5
    })([*model_body.output, *y_true])

    return Model([model_body.input, *y_true], model_loss)

def create_tiny(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2, 
                weights_path='model_data/tiny_yolo_weights.h5'):
    
    K.clear_session()

    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l],
                           num_anchors // 2, num_classes + 5)) for l in range(2)]

    model_body = TinyYOLO(image_input, num_anchors // 2, num_classes)

    print('Create Tiny YOLO3 Model with {} anchors and {} classes.'.format(
        num_anchors,
        num_classes
    ))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

        if freeze_body in [1, 2]:
            num = (20, len(model_body.layers) - 2)[freeze_body - 1]
            
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(
                num,
                len(model_body.layers)
            ))

    model_loss = Lambda(YOLOLoss, output_shape=(1,), name='yolo_loss', arguments={
        'anchors': anchors,
        'num_classes': num_classes,
        'ignore_threshold': .7,
    })([*model_body.output, *y_true])

    return Model([model_body.output, *y_true], model_loss)

def _data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''Data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0

    while True:
        image_data = []
        box_data = []

        for b in range(batch_size):
            if i == 0: np.random.shuffle(annotation_lines)
           
            image, box = get_rand_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)

            i = (i + 1) % n
        
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = PreprocessTrueBoxes(box_data, input_shape, anchors, num_classes)

        yield [image_data, *y_true], np.zeros(batch_size)
    
def _data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    if len(annotation_lines) == 0 or batch_size <= 0: return None
    return _data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()