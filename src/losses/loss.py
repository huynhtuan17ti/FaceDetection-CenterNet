import torch
from torch import nn
import torch.nn.functional as F
from .metrics import *
import numpy as np

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

class CenterLoss(nn.Module):
    def __init__(self, config):
        super(CenterLoss, self).__init__()
        self.down_stride = config['down_stride']

        self.focal_loss = modified_focal_loss
        self.iou_loss = DIOULoss
        self.l1_loss = F.l1_loss

        self.alpha = config['Loss']['alpha']
        self.beta = config['Loss']['beta']
        self.gamma = config['Loss']['gamma']

    def forward(self, pred, gt):
        pred_hm = _sigmoid(pred['hm'])
        pred_wh = pred['wh']
        pred_offset = pred['reg']
        imgs, gt_boxes, gt_classes, gt_hm, infos = gt
        gt_nonpad_mask = gt_classes.gt(0)

        cls_loss = self.focal_loss(pred_hm, gt_hm)

        wh_loss = cls_loss.new_tensor(0.)
        offset_loss = cls_loss.new_tensor(0.)
        num = 0
        for batch in range(imgs.size(0)):
            ct = infos[batch]['ct'].cuda()
            ct_int = ct.long()
            num += len(ct_int)
            batch_pos_pred_wh = pred_wh[batch, :, ct_int[:, 1], ct_int[:, 0]].view(-1)
            batch_pos_pred_offset = pred_offset[batch, :, ct_int[:, 1], ct_int[:, 0]].view(-1)

            batch_boxes = gt_boxes[batch][gt_nonpad_mask[batch]]
            wh = torch.stack([
                batch_boxes[:, 2] - batch_boxes[:, 0],
                batch_boxes[:, 3] - batch_boxes[:, 1] 
            ]).view(-1) / self.down_stride
            offset = (ct - ct_int.float()).T.contiguous().view(-1)

            wh_loss += self.l1_loss(batch_pos_pred_wh, wh, reduction='sum')
            offset_loss += self.l1_loss(batch_pos_pred_offset, offset, reduction='sum')

        regr_loss = wh_loss * self.beta + offset_loss * self.gamma
        return cls_loss * self.alpha, regr_loss / (num + 1e-6)