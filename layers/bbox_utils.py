#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import jittor as jt

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return jt.concat((boxes[:, :2] - boxes[:, 2:] / 2,     # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return jt.concat([(boxes[:, 2:] + boxes[:, :2]) / 2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2]], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = jt.minimum(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = jt.maximum(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = jt.clamp((max_xy - min_xy), min_v=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / (union + 1e-8)  # [A,B] 添加小值避免除零


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdims=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(
        0, keepdims=True)  # 0-2000
    best_truth_idx = best_truth_idx.squeeze(0)
    best_truth_overlap = best_truth_overlap.squeeze(0)
    best_prior_idx = best_prior_idx.squeeze(1)
    best_prior_overlap = best_prior_overlap.squeeze(1)
    
    # ensure best prior
    best_truth_overlap = best_truth_overlap.index_fill(0, best_prior_idx, 2)
    
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.shape[0]):
        best_truth_idx[best_prior_idx[j]] = j
        
    _th1, _th2, _th3 = threshold  # _th1 = 0.1 ,_th2 = 0.35,_th3 = 0.5

    N = (jt.sum(best_prior_overlap >= _th2) +
         jt.sum(best_prior_overlap >= _th3)) // 2
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx]         # Shape: [num_priors]
    conf = conf * (best_truth_overlap >= _th2)  # label as background

    best_truth_overlap_clone = best_truth_overlap.clone()
    add_idx = (best_truth_overlap_clone > _th1) & (best_truth_overlap_clone < _th2)
    best_truth_overlap_clone = best_truth_overlap_clone * add_idx
    
    stage2_overlap, stage2_idx = best_truth_overlap_clone.argsort(descending=True)
    stage2_overlap = stage2_overlap > _th1

    if N > 0:
        N_selected = jt.sum(stage2_overlap[:N])
        N = N_selected if N_selected < N else N
        conf[stage2_idx[:N]] += 1

    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def match_ssd(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )

    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_idx , best_prior_overlap = overlaps.argmax(dim=1)
    # [1,num_priors] best ground truth for each prior
    best_truth_idx , best_truth_overlap  = overlaps.argmax(dim=0)

    
    # ensure best prior
    best_truth_overlap[best_prior_idx] = 2.0
    
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.shape[0]):
        best_truth_idx[best_prior_idx[j]] = j
        
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx]         # Shape: [num_priors]
    conf = conf * (best_truth_overlap >= threshold)  # label as background
    
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = jt.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return jt.concat([g_cxcy, g_wh], 1)  # [num_priors,4]


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = jt.concat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * jt.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.max()
    return jt.log(jt.sum(jt.exp(x - x_max), 1, keepdims=True)) + x_max


def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = jt.zeros(scores.shape[0], dtype=jt.int64)
    if boxes.numel() == 0:
        return keep, 0
        
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    
    # sort in descending order
    v, idx = scores.sort(descending=True)
    idx = idx[:top_k]  # indices of the top-k largest vals

    count = 0
    while idx.numel() > 0:
        i = idx[0]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.shape[0] == 1:
            break
            
        idx = idx[1:]  # remove kept element from view
        
        # load bboxes of next highest vals
        xx1 = x1[idx]
        yy1 = y1[idx]
        xx2 = x2[idx]
        yy2 = y2[idx]
        
        # compute intersection
        xx1 = jt.maximum(xx1, x1[i])
        yy1 = jt.maximum(yy1, y1[i])
        xx2 = jt.minimum(xx2, x2[i])
        yy2 = jt.minimum(yy2, y2[i])
        
        w = jt.clamp(xx2 - xx1, min_v=0.0)
        h = jt.clamp(yy2 - yy1, min_v=0.0)
        inter = w * h
        
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = area[idx]
        union = (rem_areas - inter) + area[i]
        IoU = inter / (union + 1e-8)
        
        # keep only elements with an IoU <= overlap
        idx = idx[IoU <= overlap]
        
    return keep, count