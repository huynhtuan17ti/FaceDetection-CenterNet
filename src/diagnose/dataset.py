from dataset_factory.custom_dataset import FaceDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_factory.transform import Transform
import numpy as np
from utils import *
import yaml

def convert(img):
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

def show_dataset(config_path, test_path):
    config = yaml.safe_load(open(config_path))
    ds = FaceDataset(config, test_path, Transform(), mode = 'valid')
    loader = DataLoader(ds, batch_size=4, collate_fn = ds.collate_fn)
    col, row = 2, 4
    cnt = 1
    fig = plt.figure(figsize=(15, 30))

    dataiter = iter(loader)
    batch_imgs, batch_boxes, batch_classes, batch_hms, infos = dataiter.next()
    for img, bboxes, hm, classes, info in zip(batch_imgs, batch_boxes, batch_hms, batch_classes, infos):
        new_img = convert(img)
        new_img = applyBboxes(new_img, bboxes)
        fig.add_subplot(row, col, cnt)
        cnt += 1
        plt.imshow(new_img)

        new_hm = convert(hm)
        fig.add_subplot(row, col, cnt)
        cnt += 1
        plt.imshow(new_hm[:, :, 0])
    plt.show()