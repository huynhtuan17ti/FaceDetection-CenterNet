import cv2
from math import floor
import numpy as np
import config
import matplotlib.pyplot as plt

def get_img(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def convert_to_fl_list(List):
    str_list = list(map(str, List.split(' ')))
    str_list.pop(0)
    str_list.pop(0)
    str_list[3] = str_list[3][:-2]
    fl_list = [float(s) for s in str_list]
    return fl_list

def get_bboxes(label_path):
    f = open(label_path, 'r')
    bboxes = []
    for str_bbox in f:
        bbox = convert_to_fl_list(str_bbox)
        bboxes.append(bbox)
    return bboxes

def draw(path_img, bboxes):
    sample = cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB)
    sample = applyBboxes(sample, bboxes)
    plt.imshow(sample)

def applyBboxes(img, bboxes):
    for bbox in bboxes:    
        color = (220, 0, 0)
        thickness = 4
        bbox[0] = int(bbox[0])
        bbox[1] = int(bbox[1])
        bbox[2] = int(bbox[2])
        bbox[3] = int(bbox[3])
        img = cv2.UMat(img).get()
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
    return img