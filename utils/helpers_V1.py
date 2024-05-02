import os
import re
import sys
import base64
import importlib
import cv2 as cv
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import albumentations as A
import torch.nn.functional as nnf

from glob import glob

sys.path.append('..')

import OCR_API

importlib.reload(OCR_API)

from OCR_API.config import config_dict


def read_image(path_name):
    # Read image for prepeprocessing
    img = cv.imread(path_name, cv.IMREAD_GRAYSCALE)
    im_bw = cv.threshold(img, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    return im_bw

def inverted_image(img):
    # Inverted the image
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    img = np.max(img) - img
    img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
    return img

def normalize_image(img):
    # Normalize data
    img = (img - np.mean(img))/(np.std(img) + np.finfo(np.float32).eps)
    return img   

def save_images_morn(imgs):
    folder_path = config_dict['path_saved'] + 'MORN/'

    # Create folder if not exist
    if not os.path.exists(folder_path):
        os.mkdir(folder_path) 
    
    # Write images from the MORN 
    for k, img in enumerate(imgs):
        cv.imwrite(folder_path+str(k)+'.jpg', img)
    
    # Get the new path
    path_new = glob(folder_path+'*')
    path_new.sort(key=natural_keys)

    return path_new

def data_augmentation(images):
    # Data augmentation with horizontal flip and rotate
    transform = A.Compose([A.HorizontalFlip(), A.Rotate(p=0.8)])
    for k,i in enumerate(images):
        transformed = transform(image=i)
        images[k] = transformed['image']

    return images

def preprocessing_image(image, 
                        inverted=True, 
                        normalize=True, 
                        smoothing=False, 
                        sm_kernel=(7,7)):
        
    # Used the inverted function
    if inverted:
        image = inverted_image(image)

    # Used the normalize function
    if normalize:
        image = normalize_image(image)

    # Smoothing operation
    if smoothing:
        image = cv.GaussianBlur(image,sm_kernel,0)
    
    return image
        
# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text

# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def receive_image_data_url(image_data_url, type_):
    # Extract the base64-encoded image data
    encoded_data = image_data_url.split(',')[1]

    # Decode the base64-encoded image data
    binary_data = base64.b64decode(encoded_data)

    # Save the binary data to a PNG file
    path = config_dict['path_saved'] + type_ + '/'

    # Create folder
    if not os.path.exists(path):
        os.mkdir(path) 
    
    # Delete files on the folder
    else:
        files = glob(path+'*')
        if len(files) >= 5:
            for f in files:
                os.remove(f)

    # Write image
    with open(path+'captured_image.png', 'wb') as f:
        f.write(binary_data)

def get_class_pred(result, list_class, threshold=5):
    # Get all channel number
    list_id = [i for i in range(len(result.boxes.xyxy))]
    not_used_data = []
    used_data = []
    bbx = []

    # Retrive YOLO result
    boxes_xyxy = result.boxes.xyxy
    boxes_xywh = result.boxes.xywh
    boxes_conf = result.boxes.conf
    boxes_cls = result.boxes.cls

    for k, (bbx, prob) in enumerate(zip(boxes_xyxy, boxes_conf)):
        find_id = [i for i in range(len(boxes_xyxy)) 
                        if i!=k and i not in not_used_data and i not in used_data]

        # Get bounding box coordinate
        x = int(bbx[0])
        y = int(bbx[1])
        w = int(bbx[2])
        h = int(bbx[3])

        for k_i, i, prob_i in zip(find_id, boxes_xyxy[find_id], boxes_conf[find_id]):
            # Get bounding box coordinate
            x_i = int(i[0])
            y_i = int(i[1])
            w_i = int(i[2])
            h_i = int(i[3])
            
            # Check if bounding box have a similar coordinate, if between the coordinate only have 5 pixel then its the same bounding box
            if abs(x-x_i) <= threshold and abs(y-y_i) <= threshold and abs(w-w_i) <= threshold and abs(h-h_i) <= threshold:
                # Get the best probability of the class 
                if prob>prob_i:
                    used_data.append(k)
                    not_used_data.append(k_i)
                else:
                    used_data.append(k_i)
                    not_used_data.append(k)

    # Get the classification result
    used_id = [i for i in list_id if i not in not_used_data]
    cls_temp = [int(i) for i in boxes_cls[used_id].tolist()]
    cls_ = {k:i.tolist()[0] for k, (i, j) in enumerate(zip(boxes_xywh[used_id], boxes_cls[used_id]))}
    cls_ = dict(sorted(cls_.items(), key=lambda item: item[1]))
    y_pred = ''.join([list_class[cls_temp[i]] for i in cls_.keys()])
    
    return y_pred, used_id

def get_segmentation(img_full, used_id, result, inverted=False):
    # Check if there are segmentation
    if result.masks is not None:
        # Get the segmentation result
        y_pred = result.masks.data

        # Read original image
        img_full = cv.cvtColor(img_full, cv.COLOR_BGR2GRAY) 
        img_full_shape = img_full.shape

        # Resize the segmentation result into original shape
        im_temp = tf.image.resize(np.expand_dims(y_pred[used_id[0]].numpy(), -1), [img_full_shape[0], img_full_shape[1]]).numpy()
        for i in used_id[1:]:
            im_2 = tf.image.resize(np.expand_dims(y_pred[i].numpy(), -1), [img_full_shape[0], img_full_shape[1]]).numpy()
            im_temp = cv.addWeighted(im_temp, 1, im_2, 1, 0)

        # Thresholding the probability segmentation
        im_temp[im_temp>0.5] = 1
        im_temp[im_temp<0.5] = 0

        # Select only 1 channel
        if len(im_temp.shape)==3:
            im_temp = im_temp[:,:,0]
        print(im_temp.shape, img_full.shape)
        # Inverted the image and dot product with segmentation result
        if inverted:
            temp_seg = im_temp * ((255) - img_full)
        else:
            temp_seg = im_temp * ((255) - img_full)
            temp_seg = (255) - temp_seg

        temp_seg = temp_seg.astype('uint8')
        temp_seg_blur = cv.GaussianBlur(temp_seg,(7,7),0)

        # Edge detection
        im_adaptive = cv.adaptiveThreshold(temp_seg_blur,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,11,2)
        im_adaptive = cv.medianBlur(im_adaptive,5)

        # Thresholding - Edge detection and then median smoothing
        # The idea is to delete unnecessary segmentation from YOLO (because of the dataset characteristics)
        # So after this 'new segmentation' it can be an input to text recognition model
        im_th = cv.threshold(temp_seg_blur, 127, 255, cv.THRESH_OTSU)[1]
        im_adaptive_th = cv.medianBlur(im_th-im_adaptive,3)
        im_adaptive_th = cv.threshold(im_adaptive_th, 127, 255, cv.THRESH_BINARY)[1]

        return im_adaptive_th

def get_probability(output, pred):
    prob = nnf.softmax(output, dim=1)

    top_p, _ = prob.topk(1, dim = 1)
    prob_list = np.concatenate(top_p[:len(pred)].tolist()).tolist()
    prob_list = [round(i, 4) for i in prob_list]
    return prob_list

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

def get_status_image(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    img = cv.threshold(img, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    id_, count = np.unique(img, return_counts=True)
    count_dict = {k:v for k,v in zip(id_,count)}
    
    if count_dict[0] >= count_dict[255]:
        # black background
        flags = True
    else:
        # white background
        flags = False
    
    return flags