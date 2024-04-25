import os
import re
import sys
import cv2
import uvicorn
import importlib
import shutil

from glob import glob
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

sys.path.append('..')

import OCR_API

importlib.reload(OCR_API)

from OCR_API.utils.helpers import *
from OCR_API.config import config_dict
from OCR_API.text_detection import model_detection
from OCR_API.text_recognition import model_recognition
from OCR_API.text_segmentation import model_segmentation


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


# REVAMP

def segmentation(image: UploadFile = File(...)):
    save_directory = config_dict['path_saved'] + 'uploaded_images'
    os.makedirs(save_directory, exist_ok=True)
    file_path = os.path.join(save_directory, image.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    seg_class = text_seg_class(name_model='YOLO_SEG_80', 
                                file_path=file_path)

    y_pred_list = prediction_seg(seg_class)
    
    return y_pred_list

class text_seg_class(BaseModel):
    # Parameter for text segmentation
    name_model: Optional[str] = ''
    file_path : Optional[str] = ''
    smoothing : Optional[bool] = True
    sm_kernel : Optional[int] = 7

def prediction_seg(data: text_seg_class):
    model_name = data.name_model
    smoothing = data.smoothing
    sm_kernel = (data.sm_kernel, data.sm_kernel)

    y_pred_list = []
    file_path = get_used_path(data.file_path, type_='seg')

    # Initialize the model
    md = model_segmentation(model_name, smoothing=smoothing, sm_kernel=sm_kernel)
    # Prediction
    y_pred, bbx_list, conf_list, _ = md(file_path)

    # Create the json format 
    for path_file, pred, bbx, conf in zip(file_path, y_pred, bbx_list, conf_list):
        file_names = path_file.split('/')[-1]
        pred = re.findall(r'[0-9A-zA-Z]', pred)

        data_json = {
            'name': file_names,
            'prediction': pred,
            'conf': conf,
            'bbx': bbx
        }
        y_pred_list.append(data_json)

    return y_pred_list

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

# Text recognition endpoint
@app.post("/text_recognition")
async def text_recog(data: text_recog_class):
    return prediction_recog(data)

# Text detection endpoint
@app.post("/text_detection")
async def text_detect(data: text_detect_class):
    return prediction_detect(data)

# Text segmentation endpoint
@app.post("/text_segmentation")
async def text_segment(data: text_seg_class):
    return prediction_seg(data)

# Text segmentation and recognition endpoint
@app.post("/text_segment_recog")
async def text_segment_recog(data: text_seg_recog_class):
    return prediction_seg_recog(data)   

@app.post("/segmentation")
async def segmentation_async(image: UploadFile = File(...)):
    return segmentation(image)   

if __name__ == "__main__":
    uvicorn.run("api_ocr:app", host="0.0.0.0", port=2803, workers = 4)