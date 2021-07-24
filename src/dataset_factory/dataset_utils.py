import numpy as np
import cv2
import torch

def get_center(bboxes):
    # get center point for each object through its bounding box
    ct = np.array([(bboxes[..., 0] + bboxes[..., 2]) / 2,
                (bboxes[..., 1] + bboxes[..., 3]) / 2], dtype=np.float32).T
    return ct

def generate_heatmap(num_classes ,output_h, output_w, bboxes_h, bboxes_w, ct, classes):
    hm = np.zeros((len(num_classes), output_w, output_h), dtype=np.float32)
    obj_mask = torch.ones(len(classes))
    for i, cls_id in enumerate(classes):
        radius = gaussian_radius((np.ceil(bboxes_h[i]), np.ceil(bboxes_w[i])))
        radius = max(0, int(radius))
        ct_int = ct[i].astype(np.int32)
        if (hm[:, ct_int[1], ct_int[0]] == 1).sum() >= 1.:
            obj_mask[i] = 0
            continue

        draw_umich_gaussian(hm[cls_id - 1], ct_int, radius)
        if hm[cls_id-1, ct_int[1], ct_int[0]] != 1:
            obj_mask[i] = 0
            
    hm = torch.from_numpy(hm)
    obj_mask = obj_mask.eq(1)
    return hm, obj_mask

def preprocess_img_boxes(image, input_ksize, boxes=None):
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

def flip(img):
    return img[:, :, ::-1].copy()

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)
