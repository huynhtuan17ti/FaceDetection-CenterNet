import torch
from torch import nn
import torch.nn.functional as F

class IoUmetric:
    '''
        get IoU score on an image
        intersection = sum of all intersection bouding boxes on the image
        union = sum of all bouding boxes on predict and ground truth image
        IoU score = intersection/(union - intersection + esp), which esp is a value to avoid dividing by 0
    '''
    def __init__(self, threshold = None):
        super(IoUmetric, self).__init__()
        self.threshold = threshold
        self.esp = 1e-6
    
    def inter_area(self, bbox1, bbox2):
        minx, miny = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
        maxx, maxy = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
        return max(0, (minx - maxx)*(miny - maxy))

    def calc(self, bboxes1, bboxes2):
        intersection = 0
        for bbox1 in bboxes1:
            for bbox2 in bboxes2:
                intersection += self.inter_area(bbox1, bbox2)
        
        union = 0
        for bbox1 in bboxes1:
            union += (bbox1[2] - bbox1[0])*(bbox1[3] - bbox1[1])
        for bbox2 in bboxes2:
            union += (bbox2[2] - bbox2[0])*(bbox2[3] - bbox2[1])

        IoU = intersection/(union - intersection + self.esp)
        if self.threshold:
            return (IoU >= self.threshold)
        assert 0 <= IoU <= 1
        return IoU

if __name__ == '__main__':
    # testing
    bboxes1 = [[1, 1, 4, 5], [6, 3, 7, 4]]
    bboxes2 = [[3, -1, 6, 3]]
    iou = IoUmetric(threshold = 0.5)
    print(iou.calc(bboxes1, bboxes2))