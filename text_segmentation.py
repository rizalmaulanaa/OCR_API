import torch
import importlib
from ultralytics import YOLO

import utils
import OCR_API
import craft_moran_ocr

importlib.reload(utils)
importlib.reload(OCR_API)
importlib.reload(craft_moran_ocr)

from utils.helpers import *
from OCR_API.config import config_dict
from craft_moran_ocr.src.recognizer import Recognizer


def model_segmentation(model_name, smoothing=True, sm_kernel=(7,7)):
    def pred(path):
        path_pre_trained = config_dict['path_pretrain']
        model_name_split = model_name.split('_')

        y_pred = []
        bbx_list = []
        conf_list = []
        result_list = []

        if model_name_split[0]=='YOLO':
            # Get the version of the yolo
            num_epoch = model_name_split[-1]
            ck_name = path_pre_trained+'yolov8n-seg{}-0_005.pt'.format(num_epoch)
            
            for i in path:
                # Initialization
                model = YOLO(ck_name)
                # Prediction
                res = model(i)
                pred, used_id = get_class_pred(res[0], model.names)
                conf_ = [round(i,4) for i in res[0].boxes.conf[used_id].tolist()]
                bbx_ = [i for i in res[0].boxes.xyxy[used_id].to(torch.int).tolist()]

                y_pred.append(pred)
                bbx_list.append(bbx_)
                conf_list.append(conf_)
                result_list.append(res[0])
        
        elif model_name_split[0]=='MORN':
            recognizer = Recognizer()
            recognizer.load()

            # Preprocessing
            img_list = preprocessing(path, inverted=False, normalize=False, 
                                    smoothing=smoothing, sm_kernel=sm_kernel, save_img=False)
            
            # Predictione
            img_pred = [recognizer.process(img, debug=True)[2] for img in img_list]

            path_new = save_images_morn(img_pred)

            # Get the version of the yolo
            num_epoch = model_name_split[-1]
            ck_name = path_pre_trained+'yolov8n-seg{}.pt'.format(num_epoch)
            
            for i in path_new:
                # Initialization
                model = YOLO(ck_name)
                # Prediction
                res = model(i)
                pred, used_id = get_class_pred(res[0], model.names)
                conf_ = [round(i,4) for i in res[0].boxes.conf[used_id].tolist()]
                bbx_ = [i for i in res[0].boxes.xyxy[used_id].to(torch.int).tolist()]

                y_pred.append(pred)
                bbx_list.append(bbx_)
                conf_list.append(conf_)
                result_list.append(res[0])
                
        return y_pred, bbx_list, conf_list, result_list
    return pred