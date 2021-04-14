import albumentations as A
import numpy as np

class Transform(object):
    def __init__(self):
        self.aug = A.Compose([
            A.HorizontalFlip(p = 0.5),
            A.RandomBrightnessContrast(0.1, 0.1),
            #A.RGBShift(p = 0.5),
            #A.GaussNoise(),
            #A.CLAHE(),
            #A.RandomGamma(),
            #A.Blur(),
            #A.Rotate(limit=40, p = 0.5)
        ], p =1.0, bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.75, label_fields=['labels']))

    def __call__(self, img, bboxes, labels):
        augmented = self.aug(image = img, bboxes=bboxes, labels=labels)
        img, bboxes = augmented['image'], augmented['bboxes']
        bboxes = [list(bbox) for bbox in bboxes]
        bboxes = np.array(bboxes, dtype=np.float)
        return img, bboxes