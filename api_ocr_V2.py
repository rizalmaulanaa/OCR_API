import re
import sys
import uvicorn
import importlib

from glob import glob
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.append('..')

import OCR_API

importlib.reload(OCR_API)

from OCR_API.utils.helpers_V1 import *
from OCR_API.text_recognition_V2 import model_recognition
from OCR_API.text_segmentation_V2 import model_segmentation


def prediction_seg(image,
                   file_name,
                   model_name='YOLO_n_80', 
                   smoothing=True, 
                   sm_kernel=(3,3)):

    # Get status
    flag_inverted = get_status_image(image)
    
    # Inverted Image
    image = preprocessing_image(image, 
                                inverted=flag_inverted, 
                                normalize=False, 
                                smoothing=smoothing, 
                                sm_kernel=sm_kernel)

    # Initialize the model segmentation
    md = model_segmentation(model_name, preprocessing=False, smoothing=smoothing, sm_kernel=sm_kernel)

    # Prediction
    y_pred, bbx_list, bbx_n_list, conf_list, _, _ = md.prediction(image)

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

def prediction_recog(image,
                     file_name,
                     model_name='YOLO_m_60', 
                     smoothing=True, 
                     sm_kernel=(3,3)):

    # Inverted Image
    type_image_input = model_name.split('-')

    if len(type_image_input) > 1:
        # Get status
        flag_inverted = get_status_image(image)

        image = preprocessing_image(image, 
                                    inverted=flag_inverted, 
                                    normalize=False, 
                                    smoothing=smoothing, 
                                    sm_kernel=sm_kernel)

    # Initialize the model recognition
    md_recog = model_recognition(model_name=model_name, preprocessing=False, smoothing=True, sm_kernel=sm_kernel)

    y_pred, conf_list, bbx_list, bbx_n_list = md_recog.prediction(image)
    
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

def prediction_recog_double(image,
                            file_name,
                            model_name={'YOLO_m_60': False, 'YOLO_n_100-plate2': True}, 
                            smoothing=True, 
                            sm_kernel=(3,3)):

    data_json_list = []

    for md_name, v in model_name.items():
        # Inverted Image
        type_image_input = md_name.split('-')

        # Get status
        print(image.shape)
        flag_inverted = get_status_image(image, flag=v)
        image = preprocessing_image(image, 
                                    inverted=flag_inverted, 
                                    normalize=False, 
                                    smoothing=smoothing, 
                                    sm_kernel=sm_kernel)

        # Initialize the model recognition
        md_recog = model_recognition(model_name=md_name, preprocessing=False, smoothing=True, sm_kernel=sm_kernel)

        y_pred, conf_list, bbx_list, bbx_n_list = md_recog.prediction(image)
        
        # Create the json format 
        file_names = file_name.split('/')[-1]
        pred = re.findall(r'[0-9A-zA-Z]', y_pred)
        img_shape = list(image.shape)

        data_json = {
            'name': file_names,
            'prediction': pred,
            'conf': conf_list,
            'bbx': bbx_list,
            'bbxN': bbx_n_list,
            'imageShape': img_shape
        }

        data_json_list.append(data_json)

    return data_json_list

def prediction_seg_recog(image,
                     file_name,
                     model_name='YOLO_n_80', 
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
    _, bbx_list, bbx_n_list, _, used_id, result = md_seg.prediction(image_)
    img_seg = get_segmentation(image_, used_id, result, inverted=True)

    # Initialize the model recognition
    md_recog = model_recognition(model_name='MORAN', preprocessing=False, smoothing=True, sm_kernel=sm_kernel)

    y_pred, conf_list, _ = md_recog.prediction(img_seg)
    
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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

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

@app.post("/recognition_double")
async def recognition_double_async(image: UploadFile = File(...)):
    file_name = image.filename
    image = load_image_into_numpy_array(await image.read())
    md_dict = {
        'YOLO_m_60': False, 
        'YOLO_n_100-plate2': True
    }

    json_file = prediction_recog_double(image, file_name, model_name=md_dict)
    
    return json_file

if __name__ == "__main__":
    uvicorn.run("api_ocr_V2:app", host="0.0.0.0", port=2804, reload=True)