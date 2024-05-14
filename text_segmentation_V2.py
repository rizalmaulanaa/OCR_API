import torch
import importlib

from ultralytics import YOLO

import OCR_API
import craft_moran_ocr

importlib.reload(OCR_API)
importlib.reload(craft_moran_ocr)

from OCR_API.utils.helpers_V1 import *
from OCR_API.config import config_dict
from craft_moran_ocr.src.recognizer import Recognizer


class model_segmentation:
    def __init__(self, model_name, preprocessing=True, smoothing=True, sm_kernel=(7,7)):
        super(model_segmentation, self).__init__()
        self.preprocessing = preprocessing
        self.smoothing = smoothing
        self.sm_kernel = sm_kernel

        path_pre_trained = config_dict['path_pretrain']
        model_name_split = model_name.split('_')
        type_ = model_name_split[1].lower()
        num_epoch = model_name_split[-1]
        model_file_name = 'yolov8{}-seg{}.pt'.format(type_, num_epoch)

        self.ck_name = path_pre_trained + model_file_name
        self.model_name = model_name_split[0].upper()

    def prediction(self, image):
        if self.model_name=='YOLO':
            # Initialization
            model = YOLO(self.ck_name)

            # Prediction
            res = model(image)

            pred, used_id = get_class_pred(res[0], model.names)
            conf_ = [round(i,4) for i in res[0].boxes.conf[used_id].tolist()]
            bbx_ = [i for i in res[0].boxes.xywh[used_id].to(torch.int).tolist()]
            bbx_ = sorted(bbx_, key = lambda elem: elem[0]) 
            bbx_n = [i for i in res[0].boxes.xywhn[used_id].to(torch.float).tolist()]
            bbx_n = sorted(bbx_n, key = lambda elem: elem[0])
        
        elif self.model_name=='MORN':
            recognizer = Recognizer()
            recognizer.load()

            # Preprocessing
            if self.preprocessing:
                image = preprocessing_image(image, 
                                            inverted=False, 
                                            normalize=False, 
                                            smoothing=self.smoothing, 
                                            sm_kernel=self.sm_kernel)
                
            # Prediction MORN
            img_morn = recognizer.process(image, debug=True)[2]

            # Prediction YOLO
            res = model(img_morn)

            pred, used_id = get_class_pred(res[0], model.names)
            conf_ = [round(i,4) for i in res[0].boxes.conf[used_id].tolist()]
            bbx_ = [i for i in res[0].boxes.xywh[used_id].to(torch.int).tolist()]
            bbx_ = sorted(bbx_, key = lambda elem: elem[0]) 
            bbx_n = [i for i in res[0].boxes.xywhn[used_id].to(torch.float).tolist()]
            bbx_n = sorted(bbx_n, key = lambda elem: elem[0]) 
        
        else:
            pred = ''
            bbx_ = []
            conf_ = []
            res[0] = [[]]

        return pred, bbx_, bbx_n, conf_, used_id, res[0]