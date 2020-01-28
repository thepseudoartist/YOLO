import numpy as np

import tensorflow as tf
from tensorflow.python.keras import backend as K 
from tensorflow.python.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from tensorflow.python.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2

from functools import wraps

from yolo.utils import compose

@wraps(Conv2D)
def YOLOConv2D(*args, **kwargs):
    conv_kwargs = {'kernel_regularizer' : l2(5e-4)}
    conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    conv_kwargs.update(kwargs)

    return Conv2D(*args, **conv_kwargs)

def YOLOConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)

    return compose(YOLOConv2D(*args, **no_bias_kwargs), BatchNormalization(), LeakyReLU(alpha=0.1))

def ResBlock(x, num_filters, num_blocks):
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = YOLOConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)

    for i in range(num_blocks):
        y = compose(YOLOConv2D_BN_Leaky(num_filters // 2, (1, 1)), YOLOConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])

    return x

def Body(x):
    # 52 Conv2D Layered Body
    x = YOLOConv2D_BN_Leaky(32, (3, 3))(x)
    x = ResBlock(x, 64, 1)
    x = ResBlock(x, 128, 2)
    x = ResBlock(x, 256, 8)
    x = ResBlock(x, 512, 8)
    x = ResBlock(x, 1024, 4)

    return x

def  UpperLayer(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by Conv2D_Linear layer'''
    x = compose(
        YOLOConv2D_BN_Leaky(num_filters, (1, 1)),
        YOLOConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        YOLOConv2D_BN_Leaky(num_filters, (1, 1)),
        YOLOConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        YOLOConv2D_BN_Leaky(num_filters, (1, 1))
    )(x)

    y = compose(
        YOLOConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        YOLOConv2D(out_filters, (1, 1))
    )(x)

    return x, y

def YOLOBody(inputs, num_anchors, num_classes):
    '''Create YOLO3 Model in Keras'''
    
    model = Model(inputs, Body(inputs))
    x, y1 = UpperLayer(model.output, 512, num_anchors * (num_classes + 5))

    x = compose(
        YOLOConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2)
    )(x)

    x = Concatenate()([x, model.layers[152].output])
    x, y2 = UpperLayer(x, 256, num_anchors * (num_classes + 5))

    x = compose(
        YOLOConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2)
    )(x)

    x = Concatenate()([x, model.layers[92].output])
    x, y3 = UpperLayer(x, 128, num_anchors * (num_classes + 5))

    return Model(inputs, [y1, y2, y3])

def TinyYOLO(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO3 Model in Keras'''
    x1 = compose(
        YOLOConv2D_BN_Leaky(16, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        
        YOLOConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        
        YOLOConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        
        YOLOConv2D_BN_Leaky(128, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        
        YOLOConv2D_BN_Leaky(256, (3, 3))
    )(inputs)

    x2 = compose(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        YOLOConv2D_BN_Leaky(512, (3, 3)),

        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        YOLOConv2D_BN_Leaky(1024, (3, 3)),
        YOLOConv2D_BN_Leaky(256, (1, 1))
    )(x1)

    y1 = compose(
        YOLOConv2D_BN_Leaky(512, (3, 3)),
        YOLOConv2D(num_anchors * (num_classes + 5), (1, 1))
    )(x2)

    x2 = compose(
        YOLOConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2)
    )(x2)

    y2 = compose(
        Concatenate(),
        YOLOConv2D_BN_Leaky(256, (3, 3)),
        YOLOConv2D(num_anchors * (num_classes + 5), (1, 1))
    )([x2, x1])

    return Model(inputs, [y1, y2])

def YOLOHead(features, anchors, num_classes, input_shape, calc_loss=False):
    '''Convert final layer features to bounding box parameters.'''
    num_anchors = len(anchors)

    #Reshape to batch, height, width, num_anchors, box_params
    anchor_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(features)[1:3] #height, width
    
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])

    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(features))

    features = K.reshape(features, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    #Adjust predictions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(features[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(features))
    box_wh = K.exp(features[..., 2:4]) * anchor_tensor / K.cast(input_shape[::-1], K.dtype(features))

    box_confidence = K.sigmoid(features[..., 4:5])
    box_class_prob = K.sigmoid(features[..., 5:])

    if calc_loss == True:
        return grid, features, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_prob

def YOLOCorrectBoxes(box_xy, box_wh, input_shape, image_shape):
    '''Get Corrected Boxes.'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_max = box_yx + (box_hw / 2.)

    boxes = K.concatenate([
        box_mins[..., 0:1], #y min
        box_mins[..., 1:2], #x min
        box_max[..., 0:1], #y max
        box_max[..., 1:2] #x max
    ])

    #Scale boxes back to original image shape
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

def YOLOBoxesAndScores(features, anchors, num_classes, input_shape, image_shape):
    '''Process Convolution Layer Output'''
    box_xy, box_wh, box_confidence, box_class_prob = YOLOHead(features, anchors, num_classes, input_shape)

    boxes = YOLOCorrectBoxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])

    box_scores = box_confidence * box_class_prob
    box_scores = K.reshape(box_scores, [-1, num_classes])

    return boxes, box_scores

def YOLOEval(yolo_outputs, anchors, num_classes, image_shape, max_boxes=20, score_threshold=.6, iou_threshold=.5):
    '''Returns evaluated filtered boxes based on given input.'''
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []

    for i in range(num_layers):
        _boxes, _box_scores = YOLOBoxesAndScores(yolo_outputs[i], anchors[anchor_mask[i]], num_classes, input_shape, image_shape)
        
        boxes.append(_boxes)
        box_scores.append(_box_scores)

    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')

    boxes_, scores_, classes_ = [], [], []

    for i in range(num_classes):
        _class_boxes = tf.boolean_mask(boxes, mask[:, i])
        _class_boxes_scores = tf.boolean_mask(box_scores[:, i], mask[:, i])
        
        _nms_index = tf.image.non_max_suppression(_class_boxes, _class_boxes_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        _class_boxes = K.gather(_class_boxes, _nms_index)
        _class_boxes_scores = K.gather(_class_boxes_scores, _nms_index)
        _classes = K.ones_like(_class_boxes_scores, dtype='int32') * i
        
        boxes_.append(_class_boxes)
        scores_.append(_class_boxes_scores)
        classes_.append(_classes)

    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_

def PreprocessTrueBoxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to input format.
    
    true_boxes: (N, T, 5) x_min, y_min, x_max, y_max, class_id
    anchors: (N, 2)
    '''

    # Class ID must be less than NUMBER OF CLASSES
    assert(true_boxes[..., 4] < num_classes).all() 

    num_layers = len(anchors) // 3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')

    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    i = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 6}[j] for j in range(num_layers)]

    y_true = [np.zeros((i, grid_shapes[j][0], grid_shapes[j][1], len(anchor_mask[j]), num_classes + 5), dtype='float32') for j in range(num_layers)]

    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for k in range(i):
        #Discard zero rows 
        wh = boxes_wh[k, valid_mask[k]]

        if len(wh) == 0: continue
        wh = np.expand_dims(wh, -2)

        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes, - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]

        iou = intersect_area / (box_area + anchor_area - intersect_area)

        #Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.math.floor(true_boxes[k, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.math.floor(true_boxes[k, t, 1] * grid_shapes[l][0]).astype('int32')
                    p = anchor_mask[l].index(n)
                    c = true_boxes[k, t, 4].astype('int32')

                    y_true[l][k, j, i, p, 0:4] = true_boxes[k, t, 0:4]
                    y_true[l][k, j, i, p, 4] = 1
                    y_true[l][k, j, i, p, 5 + c] = 1

    return y_true

def box_iou(b1, b2):
    '''Return IOU tensor
    b1: tensor, shape=(..., 4) x, y, w, h
    b2: tensor, shape=(j, 4)

    Return: iou: tensor(..., j)
    '''

    b1 = K.expand_dims(b1, axis=-2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2 = K.expand_dims(b2, axis=0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half
    
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]

    return intersect_area / (b1_area + b2_area - intersect_area)

def YOLOLoss(args, anchors, num_classes, ignore_threshold=0.5, print_loss=False):
    '''Return YOLO Loss Tensor
    Return: loss: tensor, shape=(1,)
    '''

    num_layers = len(anchors) // 3
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]

    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shape = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]

    loss = 0

    # batch size, tensor
    batch_size = K.shape(yolo_outputs[0])[0] 
    fbatch_size = K.cast(batch_size, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_prob = y_true[l][..., 5:]
        grid, raw_pred, pred_xy, pred_wh = YOLOHead(yolo_outputs[l],
                                            anchors[anchor_mask[l]], 
                                            num_classes, 
                                            input_shape, 
                                            calc_loss=True)
        
        pred_box = K.concatenate([pred_xy, pred_wh])
        
        raw_true_xy = y_true[l][..., :2] * grid_shape[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))

        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_threshold, K.dtype(true_box)))

            return b + 1, ignore_mask
        
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < fbatch_size, loop, [0, ignore_mask])

        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * .5 * K.square(raw_true_wh - raw_pred[..., 2:4])

        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) * ignore_mask

        class_loss = object_mask * K.binary_crossentropy(true_class_prob, raw_pred[..., 5:])

        xy_loss = K.sum(xy_loss) / fbatch_size
        wh_loss = K.sum(wh_loss) / fbatch_size
        confidence_loss = K.sum(confidence_loss) / fbatch_size
        class_loss = K.sum(class_loss) / fbatch_size

        loss += xy_loss + wh_loss + confidence_loss + class_loss

        if print_loss: loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='Loss: ')

    return loss