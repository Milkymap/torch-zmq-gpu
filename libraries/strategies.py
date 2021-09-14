import cv2 
import numpy as np 
import torch as th

from torchvision import transforms as T 

from os import path, read 
from glob import glob 

def th2cv(th_image):
    red, green, blue = th_image.numpy()
    return cv2.merge((blue, green, red))

def cv2th(cv_image):
    blue, green, red = cv2.split(cv_image)
    return th.as_tensor(np.stack([red, green, blue]))

def read_image(path2image):
    cv_image = cv2.imread(path2image, cv2.IMREAD_COLOR)
    return cv_image

def save_image(cv_image, path2location):
    cv2.imwrite(path2location, cv_image)

def pull_files(target_location, extension='*'):
    return glob(path.join(target_location, extension))

def get_mapper():
    mapper = {
        'hflip': T.RandomHorizontalFlip(p=0.5),
        'vflip': T.RandomVerticalFlip(p=0.5),
        'resize': T.Resize((512, 512)),
        'center_crop': T.Compose([T.Resize((256, 256)), T.CenterCrop((128, 128))]),
        'random_rotation': T.RandomRotation(degrees=(25, 45))
    }
    return mapper 

def prepare_image(th_image):
    normalied_th_image = th_image / th.max(th_image)
    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(normalied_th_image)




