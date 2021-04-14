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

def heatmap(bbox):
    def get_coords(bbox):
        xs, ys, w, h=[],[],[],[]
        for box in bbox:
            x1, y1, x2, y2 = box
            width = (x2 - x1)
            height = (y2 - y1)
            assert width > 0 and height > 0, "find width < 0 or height < 0"
            xs.append((x1+x2)//2)
            ys.append((y1+y2)//2)
            w.append(width)
            h.append(height)
      
        return xs, ys, w, h
    
    def get_heatmap(p_x, p_y):
        # Ref: https://www.kaggle.com/diegojohnson/centernet-objects-as-points
        X1 = np.linspace(1, config.IMG_SIZE, config.IMG_SIZE)
        Y1 = np.linspace(1, config.IMG_SIZE, config.IMG_SIZE)
        [X, Y] = np.meshgrid(X1, Y1)
        X = X - floor(p_x)
        Y = Y - floor(p_y)
        D2 = X * X + Y * Y
        sigma_ = 10
        E2 = 2.0 * sigma_ ** 2
        Exponent = D2 / E2
        heatmap = np.exp(-Exponent)
        heatmap = heatmap[:, :, np.newaxis]
        return heatmap

    coors = []
    size = 5
    y_ = size
    while y_ > -size - 1:
        x_ = -size
        while x_ < size + 1:
            coors.append([x_, y_])
            x_ += 1
        y_ -= 1

    u, v, w, h = get_coords(bbox)
    
    if len(bbox) == 0:
        u = np.array([512])
        v = np.array([512])
        w = np.array([10])
        h = np.array([10])
    
    hm = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 1))
    width = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 1))
    height = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 1))

    OUT_SIZE = (config.IMG_SIZE//config.STRIDE)

    for i in range(len(u)):
        for coor in coors:
            try:
                width[int(v[i])+coor[0], int(u[i])+coor[1]] = w[i] / OUT_SIZE
                height[int(v[i])+coor[0], int(u[i])+coor[1]] = h[i] / OUT_SIZE
            except:
                pass
        heatmap = get_heatmap(u[i], v[i])
        hm[:,:] = np.maximum(hm[:,:],heatmap[:,:])
      
    hm = cv2.resize(hm, (OUT_SIZE, OUT_SIZE))[:,:,None]
    width = cv2.resize(width, (OUT_SIZE, OUT_SIZE))[:,:,None]
    height = cv2.resize(height, (OUT_SIZE, OUT_SIZE))[:,:,None]
    return hm, width, height