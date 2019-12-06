import numpy as np
import cv2
from PIL import Image
from functools import reduce

def rand(a = 0, b = 1):
    return np.random.rand() * (b - a) + a

def compose(*funcs):
    # return lambda x : reduce(lambda v, f : f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: 
                        lambda *a, **kw: 
                            g(f(*a, **kw)),
                      funcs)
    else: 
        raise ValueError("Composition of empty sequence not supported.")

def resize_img_with_pad(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_im = Image.new('RGB', size, (128, 128, 128)) # Gray
    new_im.paste(image, ((w - nw) // 2), (h - nh) // 2)

    return new_im    

def get_rand_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=0.3, sat=1.5, val=1.5, proc_img=True):
    ''' Random Processing for Real-Time Data Augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) 
                        for box in line[1:]])
    
    if not random and proc_img:
        new_im = resize_img_with_pad(image, input_shape)
    
        image_data = np.array(new_im) / 255.

        scale = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        dx, dy = (w - nw) // 2, (h - nh) // 2

        # Correct Boxes
        box_data = np.zeros((max_boxes, 5))
        
        if len(box) > 0:
            np.random.shuffle(box)
            
            if len(box) > max_boxes: box = box[:max_boxes]
            
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box)] = box
        
        return image_data, box_data

    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)

    if new_ar < 1: nh, nw = int(scale * h), int(nh * new_ar)
    else: nw, nh = int(scale * w), int(nw / new_ar)

    image = image.resize((nw, nh), Image.BICUBIC)

    dx, dy = int(rand(0, w - nw)), int(rand(0, h - nh))
    n_img = Image.new('RGB', (w, h), (128, 128, 128))
    n_img.paste(image, (dx, dy))
    image = n_img 

    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0

    image_data = hsv_to_rgb(x)

    box_data = np.zeros((max_boxes, 5))
    
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] < 0] = 0
        box[:, 2][box[:, w] > w] = w
        box[:, 2][box[:, h] > h] = h 
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]

        # Discarding Invalid Boxes
        box = box[np.logical_and(box_w > 1, box_h > 1)]

        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box
    
    return image_data, box_data

def preprocess(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)

    image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    image_padded = np.full(shape=(ih, iw, 3), fill_value=128.)

    dw, dh = (iw - nw) // 2, (ih - nh) // 2

    image_padded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_padded = np.array(image_padded, dtype='float32')
    image_padded /= 255.
    image_padded = np.expand_dims(image_padded, axis=0)

    return image_padded 
     