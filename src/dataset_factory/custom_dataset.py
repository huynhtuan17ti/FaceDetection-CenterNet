import torch
import os
from torch.utils.data import Dataset
import numpy as np 
from torchvision import transforms
from utils import get_img, get_bboxes, applyBboxes
from .dataset_utils import get_center, preprocess_img_boxes, generate_heatmap
from .transform import Transform
import json

class FaceDataset(Dataset):
    def __init__(self, config, path, transform, mode = 'train'):
        self.config = config
        self.image_path = os.path.join(path, 'images')
        self.label_path = os.path.join(path, 'label.json')
        with open(self.label_path) as json_file:
            data = json.load(json_file)
        self.data = data

        self.img_list = os.listdir(self.image_path)
            
        self.transform = transform
        self.mode = mode
        self.resize_size = (config['img_size'], config['img_size'])
        self.down_stride = config['down_stride']

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_path, self.data[index]['path'])
        bboxes = self.data[index]['bbox']

        # get image array and list of bouding boxes
        img = get_img(img_path)
        bboxes = np.array(bboxes, dtype=np.float32)

        info = {}
        num_object = len(bboxes)
        classes = [1 for _ in range(num_object)] # init class 1 for all object in image
        
        h, w, _ = img.shape
        info['raw_height'], info['raw_width'] = h, w
        info['raw_bboxes'] = bboxes

        if self.mode == 'train':
            img, bboxes = self.transform(img, bboxes, classes)
            classes = [1 for _ in range(len(bboxes))]

        img, bboxes, pad_info = preprocess_img_boxes(img, self.resize_size, boxes = bboxes)
        info['resize_height'], info['resize_width'] = img.shape[:2]
        info['pad_width'] = pad_info['pad_width']
        info['pad_height'] = pad_info['pad_height']

        ct = get_center(bboxes=bboxes)

        img = transforms.ToTensor()(img)
        bboxes = torch.from_numpy(bboxes)
        classes = torch.LongTensor(classes)
        bboxes_w, bboxes_h = bboxes[..., 2] - bboxes[..., 0], bboxes[..., 3] - bboxes[..., 1]

        # down stride
        output_h = info['resize_height'] // self.down_stride
        output_w = info['resize_width'] // self.down_stride
        bboxes_h = bboxes_h / self.down_stride
        bboxes_w = bboxes_w / self.down_stride
        ct = ct / self.down_stride
        ct[:, 0] = np.clip(ct[:, 0], 0, output_w - 1)
        ct[:, 1] = np.clip(ct[:, 1], 0, output_h - 1)

        info['gt_hm_height'], info['gt_hm_width'] = output_h, output_w

        hm, obj_mask = generate_heatmap(self.config['class_name'], output_h, output_w,
                                    bboxes_h, bboxes_w, ct, classes)
        bboxes = bboxes[obj_mask]
        classes = classes[obj_mask]
        info['ct'] = torch.tensor(ct)[obj_mask]

        return img, bboxes, classes, hm, info


    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list, hm_list, infos = zip(*data)
        assert len(imgs_list) == len(boxes_list) == len(classes_list)
        batch_size = len(boxes_list)
        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []
        pad_hm_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        max_sz = max(max_h, max_w)
    
        for i in range(batch_size):
            img = imgs_list[i]
            hm = hm_list[i]

            pad_imgs_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)(
                torch.nn.functional.pad(img, (0, int(max_sz - img.shape[2]), 0, int(max_sz - img.shape[1])), value=0.)))

            pad_hm_list.append(
                torch.nn.functional.pad(hm, (0, int(max_w//4 - hm.shape[2]), 0, int(max_h//4 - hm.shape[1])), value=0.))

        max_num = 0
        for i in range(batch_size):
            n = boxes_list[i].shape[0]
            if n > max_num:
                max_num = n
        for i in range(batch_size):
            assert boxes_list[i].shape[0] == classes_list[i].shape[0], "wrong here, {} # {}".format(boxes_list[i].shape[0], classes_list[i].shape[0])
            pad_boxes_list.append(
                torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
            pad_classes_list.append(
                torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)
        batch_hms = torch.stack(pad_hm_list)

        return batch_imgs, batch_boxes, batch_classes, batch_hms, infos