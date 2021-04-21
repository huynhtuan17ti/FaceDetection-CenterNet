# import sys
# sys.path.append('../FaceDetection-CenterNet')
import torch
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np 
import cv2
from torchvision import transforms
from utils import get_img, get_bboxes, heatmap, applyBboxes
from .dataset_utils import draw_umich_gaussian, gaussian_radius
from .transform import Transform
import json

class FaceDataset(Dataset):
    def __init__(self, path, mode = 'train', wider = False): # wider = True, use WiderFace dataset
        self.wider = wider
        if not wider:
            self.image_path = os.path.join(path, 'Image')
            self.label_path = os.path.join(path, 'Label')
            self.img_list = os.listdir(self.image_path)
        else:
            self.image_path = os.path.join(path, 'images')
            label_path = os.path.join(path, mode + '_label.json')
            with open(label_path) as json_file:
                data = json.load(json_file)
            self.data = data
            
        self.transform = Transform()
        self.mode = mode
        self.resize_size = (512, 512)
        self.down_stride = 4

    def __len__(self):
        if self.wider:
            return len(self.data)
        return len(self.img_list)

    def __getitem__(self, index):
        if not self.wider:
            img_path = os.path.join(self.image_path, self.img_list[index])
            label_path = os.path.join(self.label_path, self.img_list[index][:-4] + '.txt')
            bboxes = get_bboxes(label_path)
        else:
            img_path = os.path.join(self.image_path, self.data[index]['path'])
            bboxes = self.data[index]['bbox']
        
        # get image array and list of bboxes
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
        
        img, bboxes, pad_info = self.preprocess_img_boxes(img, self.resize_size, boxes = bboxes)
        info['resize_height'], info['resize_width'] = img.shape[:2]
        info['pad_width'] = pad_info['pad_width']
        info['pad_height'] = pad_info['pad_height']

        # get center point for each object through its bounding box
        ct = np.array([(bboxes[..., 0] + bboxes[..., 2]) / 2,
                       (bboxes[..., 1] + bboxes[..., 3]) / 2], dtype=np.float32).T

        img = transforms.ToTensor()(img)
        bboxes = torch.from_numpy(bboxes)
        classes = torch.LongTensor(classes)
        bboxes_w, bboxes_h = bboxes[..., 2] - bboxes[..., 0], bboxes[..., 3] - bboxes[..., 1]
        # print(bboxes)


        output_h, output_w = info['resize_height'] // self.down_stride, info['resize_width'] // self.down_stride
        bboxes_h, bboxes_w, ct = bboxes_h / self.down_stride, bboxes_w / self.down_stride, ct / self.down_stride
        hm = np.zeros((1, output_w, output_h), dtype=np.float32) # only one class (human face)
        
        ct[:, 0] = np.clip(ct[:, 0], 0, output_w - 1)
        ct[:, 1] = np.clip(ct[:, 1], 0, output_h - 1)

        info['gt_hm_height'], info['gt_hm_width'] = output_h, output_w
        obj_mask = torch.ones(len(classes))
        for i, cls_id in enumerate(classes):
            radius = gaussian_radius((np.ceil(bboxes_h[i]), np.ceil(bboxes_w[i])))
            radius = max(0, int(radius))
            # print('Size: ', np.ceil(bboxes_h[i]), np.ceil(bboxes_w[i]))
            # print('Radius: ', radius)
            ct_int = ct[i].astype(np.int32)
            if (hm[:, ct_int[1], ct_int[0]] == 1).sum() >= 1.:
                obj_mask[i] = 0
                continue

            draw_umich_gaussian(hm[cls_id - 1], ct_int, radius)
            if hm[cls_id-1, ct_int[1], ct_int[0]] != 1:
                obj_mask[i] = 0

        hm = torch.from_numpy(hm)
        obj_mask = obj_mask.eq(1)
        bboxes = bboxes[obj_mask]
        classes = classes[obj_mask]
        info['ct'] = torch.tensor(ct)[obj_mask]

        return img, bboxes, classes, hm, info


    def preprocess_img_boxes(self, image, input_ksize, boxes=None):
        min_side, max_side = input_ksize
        h, w, _ = image.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))
        max_sz = max(nw, nh)
        
        image_paded = cv2.copyMakeBorder(image_resized, (max_sz - nh) // 2, (max_sz - nh) - (max_sz - nh) // 2,
                                         (max_sz - nw) // 2, (max_sz - nw) - (max_sz - nw) // 2, cv2.BORDER_CONSTANT)

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + (max_sz - nw) // 2
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + (max_sz - nh) // 2
            return image_paded, boxes, {'pad_width': (max_sz - nw) // 2, 'pad_height':  (max_sz - nh) // 2}


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

if __name__ == "__main__":
    # testing
    import matplotlib.pyplot as plt
    train_ds = FaceDataset('../FaceDetection-CenterNet/WiderFaceDataset/WIDER_train', mode = 'train', wider = True)
    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=train_ds.collate_fn)
    col = 2
    row = 2
    cnt = 1
    fig = plt.figure(figsize=(15, 15))
    for data in train_loader:
        batch_imgs, batch_boxes, batch_classes, batch_hms, infos = data
        for img, bboxes, hm, classes, info in zip(batch_imgs, batch_boxes, batch_hms, batch_classes, infos):
            #print(img.shape)
            print(hm.shape)
            #print(info)
            new_img = img.permute(1, 2, 0)
            new_img = np.array(new_img, np.float32)
            new_img = applyBboxes(new_img, bboxes)
            fig.add_subplot(row, col, cnt)
            cnt += 1
            plt.imshow(new_img)
            new_hm = hm.permute(1, 2, 0)
            fig.add_subplot(row, col, cnt)
            cnt += 1
            plt.imshow(new_hm[:, :, 0])
        plt.show()
        break