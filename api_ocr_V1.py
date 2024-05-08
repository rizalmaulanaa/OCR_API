import os
import re
import sys
import cv2
import uvicorn
import importlib

from glob import glob
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

sys.path.append('..')

import OCR_API

importlib.reload(OCR_API)

from OCR_API.utils.helpers_V1 import *
from OCR_API.config import config_dict
from OCR_API.text_detection import model_detection
from OCR_API.text_recognition_V1 import model_recognition
from OCR_API.text_segmentation_V1 import model_segmentation


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

def get_used_path(file_path, type_):
    # Save image if the source from js file
    if file_path.split(';')[0] == 'data:image/png':
        receive_image_data_url(file_path, type_=type_)
        path_save = config_dict['path_saved'] +type_+ '/'
        file_path = glob(path_save+'*')

    # Get all the file in a folder
    else:
        file_path = glob(file_path+'*')

    file_path.sort(key=natural_keys)

    return file_path

class text_recog_class(BaseModel):
    # Parameter for text recognition
    name_model: Optional[str] = ''
    file_path : Optional[str] = ''
    smoothing : Optional[bool] = True
    sm_kernel : Optional[int] = 11

def prediction_recog(data: text_recog_class):
    model_name = data.name_model
    smoothing = data.smoothing
    sm_kernel = (data.sm_kernel, data.sm_kernel)

    y_pred_list = []
    file_path = get_used_path(data.file_path, type_='recog')

    # Initialize the model
    md = model_recognition(model_name, smoothing=smoothing, sm_kernel=sm_kernel)
    # Prediction
    y_pred, cls = md(file_path)

    # Create the json format 
    for path_file, pred, prob in zip(file_path, y_pred, cls):
        file_names = path_file.split('/')[-1]

        prob = get_probability(prob, pred)
        pred = re.findall(r'[0-9A-zA-Z]', pred)

        data_json = {
            'name': file_names,
            'prediction': pred,
            'conf': prob,
            'bbx': [[0,0,0,0]]
        }
        y_pred_list.append(data_json)

    return y_pred_list

class text_detect_class(BaseModel):
    # Parameter for text detection
    name_model: Optional[str] = ''
    file_path : Optional[str] = ''
    save_images: Optional[bool] = True

def prediction_detect(data: text_detect_class):
    model_name = data.name_model
    save_images = data.save_images

    y_pred_list = []
    if data.file_path[-1] == '/':
        file_path = get_used_path(data.file_path, type_='detect')
    else:
        file_path = [data.file_path]
    
    # Initialize the model
    md = model_detection(model_name, save_images)
    # Prediction
    roi, boxes = md(file_path)

    # Create the json format 
    for path_file, roi_, box in zip(file_path, roi, boxes):
        file_names = path_file.split('/')[-1]
        data_json = {
            'name': file_names,
            'images': roi_,
            'boxes': box
        }
        y_pred_list.append(data_json)

    return y_pred_list

class text_seg_recog_class(BaseModel):
    # Parameter for text segmentation and recognition
    model_seg: Optional[str] = ''
    model_recog: Optional[str] = ''
    file_path : Optional[str] = ''
    smoothing : Optional[bool] = True
    sm_kernel : Optional[int] = 11
    inverted : Optional[bool] = False
    result_segmentation: Optional[list] = None

def prediction_seg_recog(data:text_seg_recog_class):
    model_seg = data.model_seg
    model_recog = data.model_recog
    smoothing = data.smoothing
    inverted = data.inverted
    sm_kernel = (data.sm_kernel, data.sm_kernel)
    result_segmentation = data.result_segmentation

    y_pred_list = []
    file_path = get_used_path(data.file_path, type_='seg_recog')

    # Initialize the model
    md_seg = model_segmentation(model_seg)
    # Prediction
    _, used_channel, results = md_seg(file_path)

    # Create new folder
    if 'clean' in result_segmentation:
        type_new = 'seg_clean'
        path = config_dict['path_saved'] + type_new + '/'

        if not os.path.exists(path):
            os.mkdir(path)

        # Delete all the file in a folder
        else:
            files = glob(path+'*')
            for f in files:
                os.remove(f)

    if 'raw' in result_segmentation:
        type_new = 'seg_recog'
        path = config_dict['path_saved'] + type_new + '/'

        if not os.path.exists(path):
            os.mkdir(path)

        # Delete all the file in a folder
        else:
            files = glob(path+'*')
            for f in files:
                os.remove(f)
    
    if 'inverted' in result_segmentation:
        type_new = 'seg_inverted'
        path = config_dict['path_saved'] + type_new + '/'

        if not os.path.exists(path):
            os.mkdir(path)

        # Delete all the file in a folder
        else:
            files = glob(path+'*')
            for f in files:
                os.remove(f)

    # Write segmentation result
    for img_path, used_id, result in zip(file_path, used_channel, results):
        # Condition for clean segmentation
        write_segmentation(img_path, used_id, result, 
                    inverted=inverted, result_segmentation=result_segmentation)
        # write_image_segmentation(img_path, used_id, result, inverted=inverted)

    # Get new file path
    if 'clean' == result_segmentation[0]:
        path = config_dict['path_saved'] + 'seg_clean/'
    elif 'raw' == result_segmentation[0]:
        path = config_dict['path_saved'] + 'seg_recog/'
    else:
        path = config_dict['path_saved'] + 'seg_inverted/'

    path_new = get_used_path(path, type_=type_new)

    # Initialize the model
    md_recog = model_recognition(model_recog, smoothing=smoothing, sm_kernel=sm_kernel)
    # Prediction
    y_pred = md_recog(path_new)

    # Create the json format 
    for path_file, pred in zip(path_new, y_pred):
        file_names = path_file.split('/')[-1]
        data_json = {
            'name': file_names,
            'result': pred
        }
        y_pred_list.append(data_json)

    return y_pred_list


# REVAMP ==============================================================================================

def prediction_seg(image,
                   file_name,
                   model_name='YOLO_SEG_80', 
                   smoothing=True, 
                   sm_kernel=(3,3)):

    # Initialize the model
    md = model_segmentation(model_name, smoothing=smoothing, sm_kernel=sm_kernel)

    # Get status
    flag_inverted = get_status_image(image)
    
    # Inverted Image
    image = preprocessing_image(image, 
                                inverted=flag_inverted, 
                                normalize=False, 
                                smoothing=smoothing, 
                                sm_kernel=sm_kernel)

    # Prediction
    y_pred, bbx_list, conf_list, _, _ = md(image)

    # Create the json format 
    file_names = file_name.split('/')[-1]
    pred = re.findall(r'[0-9A-zA-Z]', y_pred)
    img_shape = list(image.shape)

    data_json = [{
        'name': file_names,
        'prediction': pred,
        'conf': conf_list,
        'bbx': bbx_list,
        'imageShape': img_shape
    }]
    return data_json

def prediction_recog(image,
                     file_name,
                     model_name='YOLO_40-ocr', 
                     smoothing=True, 
                     sm_kernel=(3,3)):
    
    # Get status
    flag_inverted = get_status_image(image)

    # Inverted Image
    image_ = preprocessing_image(image, 
                                 inverted=flag_inverted, 
                                 normalize=False, 
                                 smoothing=smoothing, 
                                 sm_kernel=sm_kernel)

    # Initialize the model recognition
    md_recog = model_recognition(model_name=model_name, preprocessing=False, smoothing=True, sm_kernel=sm_kernel)

    y_pred, conf_list, bbx_list, bbx_n_list = md_recog(image_)
    
    # Create the json format 
    file_names = file_name.split('/')[-1]
    pred = re.findall(r'[0-9A-zA-Z]', y_pred)
    img_shape = list(image.shape)

    data_json = [{
        'name': file_names,
        'prediction': pred,
        'conf': conf_list,
        'bbx': bbx_list,
        'bbxN': bbx_n_list,
        'imageShape': img_shape
    }]
    return data_json

def prediction_seg_recog(image,
                     file_name,
                     model_name='YOLO_SEG_80', 
                     smoothing=True, 
                     sm_kernel=(3,3)):
    
    # Initialize the model segmentation
    md_seg = model_segmentation(model_name, smoothing=smoothing, sm_kernel=sm_kernel)
    
    # Get status
    flag_inverted = get_status_image(image)

    # Inverted Image
    image_ = preprocessing_image(image, 
                                 inverted=flag_inverted, 
                                 normalize=False, 
                                 smoothing=smoothing, 
                                 sm_kernel=sm_kernel)

    # Prediction segmentation
    _, bbx_list, _, used_id, result = md_seg(image_)
    img_seg = get_segmentation(image_, used_id, result, inverted=True)

    # Initialize the model recognition
    md_recog = model_recognition(model_name='MORAN', preprocessing=False, smoothing=True, sm_kernel=sm_kernel)

    y_pred, conf_list, _ = md_recog(img_seg)
    
    # Create the json format 
    file_names = file_name.split('/')[-1]
    pred = re.findall(r'[0-9A-zA-Z]', y_pred)
    img_shape = list(image.shape)

    data_json = [{
        'name': file_names,
        'prediction': pred,
        'conf': conf_list,
        'bbx': bbx_list,
        'imageShape': img_shape
    }]
    return data_json

# Text segmentation endpoint
@app.post("/segmentation")
async def segmentation_async(image: UploadFile = File(...)):
    file_name = image.filename
    image = load_image_into_numpy_array(await image.read())
    json_file = prediction_seg(image, file_name)
    
    return json_file

@app.post("/recognition")
async def recognition_async(image: UploadFile = File(...)):
    file_name = image.filename
    image = load_image_into_numpy_array(await image.read())
    json_file = prediction_recog(image, file_name)
    
    return json_file

if __name__ == "__main__":
    uvicorn.run("api_ocr_V1:app", host="0.0.0.0", port=2804, workers = 4)