import torch
from torch import nn
import torch.nn.functional as F
from config import Config

cfg = Config()

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap

def inference(net, img, infos, topK=40, return_hm=False, th=None):
    cfg = Config()
    pred = net(img)
    pred_hm = _sigmoid(pred['hm'])
    pred_wh = pred['wh']
    pred_offset = pred['reg']

    _, _, h, w = img.shape
    b, c, output_h, output_w = pred_hm.shape

    pred_hm = pool_nms(pred_hm)

    scores, index, clses, ys, xs = topk_score(pred_hm, K=topK)

    reg = gather_feature(pred_offset, index, use_transform=True)
    reg = reg.reshape(b, topK, 2)
    xs = xs.view(b, topK, 1) + reg[:, :, 0:1]
    ys = ys.view(b, topK, 1) + reg[:, :, 1:2]

    wh = gather_feature(pred_wh, index, use_transform=True)

    wh = wh.reshape(b, topK, 2)
        

    clses = clses.reshape(b, topK, 1).float()
    scores = scores.reshape(b, topK, 1)

    half_w, half_h = wh[..., 0:1] / 2, wh[..., 1:2] / 2
    bboxes = torch.cat([xs - half_w, ys - half_h, xs + half_w, ys + half_h], dim=2)

    
    detects = []
    for batch in range(b):
        mask = scores[batch].gt(cfg.score_th if th is None else th)

        batch_boxes = bboxes[batch][mask.squeeze(-1), :]
        max_sz = max(infos[batch]['raw_width'], infos[batch]['raw_height'])
        batch_boxes[:, [0, 2]] *= max_sz / output_w
        batch_boxes[:, [1, 3]] *= max_sz / output_h
        batch_boxes[:, [0, 2]] -= infos[batch]['pad_width']
        batch_boxes[:, [1, 3]] -= infos[batch]['pad_height']

        batch_scores = scores[batch][mask]

        batch_clses = clses[batch][mask]
        batch_clses = [cfg.CLASSES_NAME[int(cls.item())] for cls in batch_clses]

        detects.append([batch_boxes, batch_scores, batch_clses, pred_hm[batch] if return_hm else None])
    return detects

def pool_nms(hm, pool_size=3):
    pad = (pool_size - 1) // 2
    hm_max = F.max_pool2d(hm, pool_size, stride=1, padding=pad)
    keep = (hm_max == hm).float()
    return hm * keep

def topk_score(scores, K):
    batch, channel, height, width = scores.shape

    # get topk score and its index in every H x W(channel dim) feature map
    topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # get all topk in in a batch
    topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
    # div by K because index is grouped by K(C x K shape)
    topk_clses = (index / K).int()
    topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
    topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
    topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs