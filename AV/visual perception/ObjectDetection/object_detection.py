import os
import cv2 
import torch
import numpy as np
from ultralytics import YOLO

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# run on GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device {device}')




class ObjectDetection: 
    def __init__(self) -> None:
        self.model = YOLO('yolov8x.pt')  # load an official model

    def detect(self, img: np.ndarray, save_path : str) -> None:
        canvas = self.model(img, presist=True)[0].plot()        
        
        cv2.imwrite(save_path, canvas)

class ObjectTracking: 

    def __init__(self) : 
        self.model = YOLO('yolov8x.pt')
    
    def track(self, img: np.ndarray, save_path: str) -> None:
        canvas = self.model.track(img, presist=True)[0].plot()
        
        # save image
        cv2.imwrite(save_path, canvas)
