import cv2 
import numpy as np 
import torch as th

from torchvision import transforms as T 

from os import path, read 
from glob import glob 

def th2cv(th_image):
    red, green, blue = th_image
    return cv2.merge((blue, green, red))

def cv2th(cv_image):
    blue, green, red = cv2.split(cv_image)
    return th.as_tensor(np.stack([red, green, blue]))

def read_image(path2image):
    cv_image = cv2.imread(path2image, cv2.IMREAD_COLOR)
    return cv_image

def pull_files(target_location, extension='*'):
    return glob(path.join(target_location, extension))






