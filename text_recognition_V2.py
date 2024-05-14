import sys
import torch
import importlib

from ultralytics import YOLO

sys.path.append("..")

import OCR_API
import craft_moran_ocr

importlib.reload(OCR_API)
importlib.reload(craft_moran_ocr)

from OCR_API.utils.helpers_V1 import *
from OCR_API.config import config_dict
from craft_moran_ocr.src.recognizer import Recognizer


class model_recognition:
    def __init__(self, model_name, preprocessing=True, smoothing=True, sm_kernel=(7,7)):
        super(model_recognition, self).__init__()
        self.preprocessing = preprocessing
        self.smoothing = smoothing
        self.sm_kernel = sm_kernel

        path_pre_trained = config_dict['path_pretrain']
        model_name_split = model_name.split('_')
        type_ = model_name_split[1].lower()
        num_epoch = model_name_split[-1]
        model_file_name = 'yolov8{}-{}.pt'.format(type_, num_epoch)

        self.ck_name = path_pre_trained + model_file_name
        self.model_name = model_name_split[0].upper()

    def prediction(self, image):
        # Using the model MORAN
        if self.model_name=='MORAN':
            # Initialize the model
            recognizer = Recognizer()
            recognizer.load()

            # Preprocessing
            if self.preprocessing:
                image = preprocessing_image(image, inverted=True, normalize=False, 
                                            smoothing=self.smoothing, sm_kernel=self.sm_kernel)
            
            # Prediction
            out = recognizer.process(image)
            y_pred = out[0]
            cls = out[-1]

            conf_ = get_probability(cls, y_pred)
            bbx_ = []
            bbx_n = []

        elif self.model_name=='YOLO':
            # Initialization
            model = YOLO(self.ck_name)

            # Prediction
            res = model(image)

            y_pred, used_id = get_class_pred(res[0], model.names)
            conf_ = [round(i,4) for i in res[0].boxes.conf[used_id].tolist()]
            bbx_ = [i for i in res[0].boxes.xywh[used_id].to(torch.float).tolist()]
            bbx_ = sorted(bbx_, key = lambda elem: elem[0]) 
            bbx_n = [i for i in res[0].boxes.xywhn[used_id].to(torch.float).tolist()]
            bbx_n = sorted(bbx_n, key = lambda elem: elem[0]) 
        
        else:
            y_pred = ''
            conf_ = []
            bbx_ = []
            bbx_n = []

        return y_pred, conf_, bbx_, bbx_n