import sys
import easyocr
import importlib
import pytesseract

from ultralytics import YOLO

sys.path.append("..")

import utils
import OCR_API
import crnn_pytorch
import craft_moran_ocr
import DigitRecognition

importlib.reload(utils)
importlib.reload(OCR_API)
importlib.reload(crnn_pytorch)
importlib.reload(craft_moran_ocr)
importlib.reload(DigitRecognition)

from utils.helpers_V1 import *
from OCR_API.config import config_dict
from crnn_pytorch.src.predict import *
from DigitRecognition.process_image import *
from craft_moran_ocr.src.recognizer import Recognizer


def model_recognition(model_name, preprocessing=True, smoothing=True, sm_kernel=(11,11)):
    def pred(image):
        path_pre_trained = config_dict['path_pretrain']
        model_name_split = model_name.split('_')

        # Using the model CRNN
        if model_name=='MORAN':
            # Initialize the model
            recognizer = Recognizer()
            recognizer.load()
            
            # Preprocessing
            if preprocessing:
                image = preprocessing_image(image, 
                                            inverted=True, 
                                            normalize=False, 
                                            smoothing=smoothing, 
                                            sm_kernel=sm_kernel)
            
            # Prediction
            out = recognizer.process(image)
            y_pred = out[0]
            cls = out[-1]

            conf_ = get_probability(cls, y_pred)

        elif model_name_split[0]=='YOLO':
            # Get the version of the yolo
            num_epoch = model_name_split[-1]
            ck_name = path_pre_trained+'yolov8m-{}.pt'.format(num_epoch)

            model = YOLO(ck_name)
            res = model(image)
            y_pred, used_id = get_class_pred(res[0], model.names)
            conf_ = [round(i,4) for i in res[0].boxes.conf[used_id].tolist()]
            bbx_ = [i for i in res[0].boxes.xywh[used_id].to(torch.int).tolist()]
            bbx_ = sorted(bbx_, key = lambda elem: elem[0]) 

        if len(conf_) == 0:
            conf_ = [0.0]*len(y_pred)
            
        if len(bbx_) == 0:
            bbx_ = [[0,0,0,0]]

        return y_pred, conf_, bbx_
    return pred