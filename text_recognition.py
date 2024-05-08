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

from utils.helpers import *
from OCR_API.config import config_dict
from crnn_pytorch.src.predict import *
from DigitRecognition.process_image import *
from craft_moran_ocr.src.recognizer import Recognizer


def model_recognition(model_name, smoothing=True, sm_kernel=(11,11)):
    def pred(path):
        y_pred, cls = [], []
        path_pre_trained = config_dict['path_pretrain']
        model_name_split = model_name.split('_')

        # Using the model tesseract
        if model_name=='EasyOCR':
            reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
            
            for i in path:
                result = reader.readtext(i)
                if len(result) >= 1:
                    result = re.sub('\'', '', result[0][1])
                    result = ''.join(result.split(' '))
                else:
                    result = ''

                y_pred.append(result)

        elif model_name=='tesseract':
            # Preprocessing
            img_list = preprocessing(path, inverted=False, normalize=False, 
                                    smoothing=smoothing, sm_kernel=sm_kernel, save_img=False)
            
            # Prediction
            y_pred = [pytesseract.image_to_string(i) for i in img_list]

        # Using the model CRNN
        elif model_name=='CRNN':
            ck_name = path_pre_trained+'crnn_synth90k.pt'
            
            # Prediction
            y_pred = main_predict(ck_name, path)
            y_pred = [''.join(i) for i in y_pred]
        
        # Using the model CNN Based
        elif model_name=='CNN_Based':
            
            for i in path:
                # Prediction
                y_ = get_output_image(i)[1]
                y_ = ''.join(y_)
                y_pred.append(y_)

        # Using the model MORAN
        elif model_name=='MORAN':
            # Initialize the model
            recognizer = Recognizer()
            recognizer.load()
            
            # Preprocessing
            img_list = preprocessing(path, inverted=True, normalize=False, 
                                    smoothing=smoothing, sm_kernel=sm_kernel, save_img=False)
            
            # Prediction
            for img in img_list:
                out = recognizer.process(img)
                prob = get_probability(out[-1], out[0])
                y_pred.append(out[0])
                cls.append(prob)

        # Using the model YOLO
        elif model_name_split[0]=='YOLO':
            # Get the version of the yolo
            num_epoch = model_name_split[-1]
            ck_name = path_pre_trained+'yolov8n-{}.pt'.format(num_epoch)
            
            for i in path:
                # Initialize the model
                model = YOLO(ck_name)
                # Prediction
                res = model(i)

                # Get the classification result
                pred, used_id = get_class_pred(res[0], model.names)
                conf_ = [round(i,4) for i in res[0].boxes.conf[used_id].tolist()]
                bbx_ = [i for i in res[0].boxes.xywh[used_id].to(torch.int).tolist()]
                bbx_ = sorted(bbx_, key = lambda elem: elem[0]) 
                cls.append(conf_)
                # cls_temp = [str(int(i)) for i in result.boxes.cls.tolist()]
                # cls_ = {k:i.tolist()[0] for k, (i, j) in enumerate(zip(result.boxes.xywh, result.boxes.cls))}
                # cls_ = dict(sorted(cls_.items(), key=lambda item: item[1]))
                # y_temp = ''.join([cls_temp[i] for i in cls_.keys()])
                y_pred.append(pred)

        # Using the model MORN + YOLO
        elif model_name=='MORN_YOLO':
            # Initialize the model
            ck_name = path_pre_trained+'best.pt' # best
            recognizer = Recognizer()
            recognizer.load()

            # Preprocessing
            img_list = preprocessing(path, inverted=False, normalize=False, 
                                    smoothing=smoothing, sm_kernel=sm_kernel, save_img=False)
            
            # Predictione
            img_pred = [recognizer.process(img, debug=True)[2] for img in img_list]

            path_new = save_images_morn(img_pred)

            for i in path_new:
                # Initialize the model
                model = YOLO(ck_name)
                # Prediction
                result = model(i)[0]

                # Get the classification result
                cls_temp = [str(int(i)) for i in result.boxes.cls.tolist()]
                cls_ = {k:i.tolist()[0] for k, (i, j) in enumerate(zip(result.boxes.xywh, result.boxes.cls))}
                cls_ = dict(sorted(cls_.items(), key=lambda item: item[1]))
                y_temp = ''.join([cls_temp[i] for i in cls_.keys()])
                y_pred.append(y_temp)

        if len(cls) == 0:
            cls = [[0.0]*len(i) for i in y_pred]

        return y_pred, cls
    return pred