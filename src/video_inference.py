import torch
import pafy
import numpy as np
import yaml
from time import time
from torchvision import transforms 
import cv2
from models.resnet_dcn.get_model import create_model, load_model
from models.resnet_dcn.utils import inference
from utils import applyBboxes


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

    return image_paded, {'raw_height': h, 'raw_width': w, 'pad_width': (largest_side - w) // 2, 'pad_height':  (largest_side - h) // 2}

def expand_bboxes(cfg, bboxes, w, h):
    expand_w = (bboxes[..., 2] - bboxes[..., 0])*cfg['expand_percent']
    expand_h = (bboxes[..., 3] - bboxes[..., 1])*cfg['expand_percent']

    bboxes[..., 0] = bboxes[..., 0] - expand_w
    bboxes[..., 2] = bboxes[..., 2] + expand_w
    bboxes[..., 1] = bboxes[..., 1] - expand_h
    bboxes[..., 3] = bboxes[..., 3] + expand_h

    bboxes[..., 0] = torch.clamp(bboxes[..., 0], 0, w-1)
    bboxes[..., 2] = torch.clamp(bboxes[..., 2], 0, w-1)

    bboxes[..., 1] = torch.clamp(bboxes[..., 1], 0, h-1)
    bboxes[..., 3] = torch.clamp(bboxes[..., 3], 0, h-1)

    return bboxes

def show_img(cfg, frame, boxes, clses, scores):
    boxes, scores = [i.cpu() for i in [boxes, scores]]
    h, w, _ = frame.shape
    if cfg['expand_percent'] > 0.0:
        boxes = expand_bboxes(cfg, boxes, w, h)

    boxes = boxes.long()
    frame = applyBboxes(frame, boxes)

    boxes = boxes.tolist()
    scores = scores.tolist()
    
    return frame


class RealtimeDetection:
    def __init__(self, config, url, out_file = 'labeled_video.mp4'):
        self._URL = url
        self.out_file = out_file
        self.device = torch.device('cpu' if config['gpu'] < 0 else 'cuda:%s' % config['gpu'])
        self.config = config
        self.net = self.load_model()
    
    def get_video_from_url(self):
        play = pafy.new(self._URL).streams[-1]
        assert play is not None, "url doesn't exist"
        return cv2.VideoCapture(play.url)

    def predict_per_frame(self, frame):
        paded_frame, info = preprocess_img(frame, (self.config['img_size'], self.config['img_size']))
        frames = [frame]
        infos = [info]

        input = transforms.ToTensor()(paded_frame) # convert to torch tensor
        input = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input)
        inputs = input.unsqueeze(0).to(self.device)

        detects = inference(self.config, self.net, inputs, infos, topK = 50, return_hm = False, th=0.3)

        boxes = detects[0][0]
        scores = detects[0][1]
        clses = detects[0][2]

        frame = frames[0]

        return show_img(self.config, frame, boxes, clses, scores)

    def load_model(self):
        net = create_model(self.config['model_name'], self.config['CNN']['heads'], self.config['CNN']['head_conv']).to(self.device)
        net.load_state_dict(torch.load(self.config['save_path'] + '/' + self.config['save_model']))
        print('Load model successfully!')
        return net

    def __call__(self):
        player = self.get_video_from_url()
        assert player.isOpened()
        x_size = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_size = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_size, y_size))
        while True:
            start_time =  time()
            ret, frame = player.read()
            if ret is None or frame is None:
                break
            frame = self.predict_per_frame(frame)
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)
            print(f'Frames Per Second: {fps}')
            out.write(frame)

if __name__ == '__main__':
    config = yaml.safe_load(open('config.yaml'))
    video = RealtimeDetection(config, "https://www.youtube.com/watch?v=nW948Va-l10")
    video()