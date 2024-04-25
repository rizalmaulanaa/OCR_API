import os
import sys
import importlib
import cv2 as cv

from glob import glob

sys.path.append("..")

import OCR_API
import craft_moran_ocr

importlib.reload(OCR_API)
importlib.reload(craft_moran_ocr)

from OCR_API.config import config_dict
from craft_moran_ocr.src.detector import Detector


def model_detection(model_name, save_images=True):
    def pred(path):
        path_save_images = config_dict['path_saved'] + 'pred_detection/'
        roi_list = []
        boxes_list = []

        for path_ in path:
            # Using model CRAFT
            if model_name == 'CRAFT':
                # Read image
                image = cv.imread(path_)

                # Initialize model
                detector = Detector()
                detector.load()
                # Predict
                roi, boxes, _, _ = detector.process(image)

            if save_images:
                # Get new folder name 
                name_file = path_.split('/')[-1]
                id_file = name_file.split('.')[0]
                folder_path_new = path_save_images+id_file

                # Create new folder
                if not os.path.exists(folder_path_new):
                    os.mkdir(folder_path_new+'/') 
                
                # Delete all file in a folder
                else:
                    files = glob(folder_path_new+'/*')
                    for f in files:
                        os.remove(f)
                
                # Write all roi 
                for k, img in enumerate(roi):
                    name_ = folder_path_new + '/' + str(k) + '.jpg'
                    cv.imwrite(name_, img)

            # Convert into list
            roi = [i.tolist() for i in roi]
            boxes = [i.tolist() for i in list(boxes)]

            roi_list.append(roi)
            boxes_list.append(boxes)
        
        return roi_list, boxes_list
    return pred