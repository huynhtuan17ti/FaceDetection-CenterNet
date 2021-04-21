from config import Config
from models.resnet_dcn.get_model import create_model, load_model
from models.resnet_dcn.utils import inference
import torch
import cv2
import os
from utils import get_img, applyBboxes
from torchvision import transforms 
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def load_model(cfg):
    net = create_model(cfg.model_name, cfg.heads, cfg.head_conv).to(cfg.device_inference)
    net.load_state_dict(torch.load(cfg.save_path + '/' + cfg.save_model))
    print('Load model successfully!')
    return net

def preprocess_img(image, input_ksize):
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

    # print("raw height: {}, raw width: {}".format(h, w))
    # print("pad_height: {}, pad_width: {}".format((largest_side - w) // 2, (largest_side - h) // 2))
    return image_paded, {'raw_height': h, 'raw_width': w, 'pad_width': (largest_side - w) // 2, 'pad_height':  (largest_side - h) // 2}

def expand_bboxes(bboxes, w, h):
    expand_w = (bboxes[..., 2] - bboxes[..., 0])*cfg.expend_percent
    expand_h = (bboxes[..., 3] - bboxes[..., 1])*cfg.expend_percent

    bboxes[..., 0] = bboxes[..., 0] - expand_w
    bboxes[..., 2] = bboxes[..., 2] + expand_w
    bboxes[..., 1] = bboxes[..., 1] - expand_h
    bboxes[..., 3] = bboxes[..., 3] + expand_h

    bboxes[..., 0] = torch.clamp(bboxes[..., 0], 0, w-1)
    bboxes[..., 2] = torch.clamp(bboxes[..., 2], 0, w-1)

    bboxes[..., 1] = torch.clamp(bboxes[..., 1], 0, h-1)
    bboxes[..., 3] = torch.clamp(bboxes[..., 3], 0, h-1)

    return bboxes

def show_img(img, boxes, clses, scores, save_path):
    boxes, scores = [i.cpu() for i in [boxes, scores]]
    h, w, _ = img.shape
    if cfg.expand_bbox:
        boxes = expand_bboxes(boxes, w, h)

    boxes = boxes.long()
    img = applyBboxes(img, boxes)

    boxes = boxes.tolist()
    scores = scores.tolist()
    
    for i in range(len(boxes)):
        plt.text(x=boxes[i][0], y=boxes[i][1], s='{:.4f}'.format(scores[i]), wrap=True, size=10,
                 bbox=dict(facecolor="r", alpha=0.5))
    
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

if __name__ == '__main__':
    cfg = Config()
    net = load_model(cfg)
    net.eval()

    img_list = os.listdir(cfg.test_path)
    print('Found {} images! Starting predict'.format(len(img_list)))
    for num, img_name in enumerate(img_list):
        print('Predicting image {} ...'.format(num))
        img_path = os.path.join(cfg.test_path, img_name)
        img = get_img(img_path)

        img_paded, info = preprocess_img(img, cfg.resize_size)
        imgs = [img]
        infos = [info]

        input = transforms.ToTensor()(img_paded)
        input = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input)
        inputs = input.unsqueeze(0).to(cfg.device_inference)
        print('Preprocess done !')

        detects = inference(net, inputs, infos, topK = 50, return_hm = False, th=0.2)

        print('Done! Prepare to show the result ...')
        for img_idx in range(len(detects)):
            fig = plt.figure(figsize=(10, 10))

            boxes = detects[img_idx][0]
            scores = detects[img_idx][1]
            clses = detects[img_idx][2]
            hm = detects[img_idx][3]

            # used to check heat map
            '''
            # hm = hm.detach().cpu().numpy()
            hm = hm.cpu()
            new_hm = hm.permute(1, 2, 0)
            new_hm = new_hm.detach().numpy()
            print(hm.shape)
            plt.imshow(new_hm[:, :, 0])
            plt.show()
            '''

            img = imgs[img_idx]

            show_img(img, boxes, clses, scores, os.path.join(cfg.save_img_path, 'sample{}.jpg'.format(num)))
            plt.show()


    
