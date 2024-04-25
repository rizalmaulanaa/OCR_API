import torch
import importlib
from ultralytics import YOLO

import utils
import OCR_API
import craft_moran_ocr

importlib.reload(utils)
importlib.reload(OCR_API)
importlib.reload(craft_moran_ocr)

from utils.helpers_V1 import *
from OCR_API.config import config_dict
from craft_moran_ocr.src.recognizer import Recognizer


def model_segmentation(model_name, 
                       smoothing=True, 
                       sm_kernel=(7,7)):

    def pred(image):
        path_pre_trained = config_dict['path_pretrain']
        model_name_split = model_name.split('_')

        # Get the version of the yolo
        num_epoch = model_name_split[-1]
        ck_name = path_pre_trained+'yolov8n-seg{}.pt'.format(num_epoch)

        # Initialization
        model = YOLO(ck_name)

        if model_name_split[0]=='YOLO':
            # Prediction
            res = model(image)
            pred, used_id = get_class_pred(res[0], model.names)
            conf_ = [round(i,4) for i in res[0].boxes.conf[used_id].tolist()]
            bbx_ = [i for i in res[0].boxes.xywh[used_id].to(torch.int).tolist()]
            bbx_ = sorted(bbx_, key = lambda elem: elem[0]) 
        
        elif model_name_split[0]=='MORN':
            recognizer = Recognizer()
            recognizer.load()

            # Preprocessing
            image = preprocessing_image(image, 
                                        inverted=False, 
                                        normalize=False, 
                                        smoothing=smoothing, 
                                        sm_kernel=sm_kernel)
            
            # Prediction morn
            img_morn = recognizer.process(image, debug=True)[2]

            # Prediction yolo
            res = model(img_morn)
            pred, used_id = get_class_pred(res[0], model.names)
            conf_ = [round(i,4) for i in res[0].boxes.conf[used_id].tolist()]
            bbx_ = [i for i in res[0].boxes.xywh[used_id].to(torch.int).tolist()]
            bbx_ = sorted(bbx_, key = lambda elem: elem[0]) 
        
        else:
            pred = ''
            bbx_ = []
            conf_ = []
            res[0] = [[]]
                
        return pred, bbx_, conf_, res[0]
    return pred