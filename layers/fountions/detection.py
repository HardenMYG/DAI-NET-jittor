#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import jittor as jt
from ..bbox_utils import decode, nms

class Detect(jt.Module):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, cfg):
        super(Detect, self).__init__()
        self.num_classes = cfg.NUM_CLASSES
        self.top_k = cfg.TOP_K
        self.nms_thresh = cfg.NMS_THRESH  # 用于决定哪些边界框重叠
        self.conf_thresh = cfg.CONF_THRESH  # 用于筛选置信度高的结果
        self.variance = cfg.VARIANCE
        self.nms_top_k = cfg.NMS_TOP_K

    def execute(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4] 
        """
        num = loc_data.shape[0]
        num_priors = prior_data.shape[0]

        conf_preds = conf_data.view(
            num, num_priors, self.num_classes).transpose(1, 2)
        batch_priors = prior_data.view(-1, num_priors,
                                       4).expand(num, num_priors, 4)
        batch_priors = batch_priors.reshape(-1, 4)

        decoded_boxes = decode(loc_data.reshape(-1, 4),
                               batch_priors, self.variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        output = jt.zeros((num, self.num_classes, self.top_k, 5))

        for i in range(num):
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl] > self.conf_thresh
                scores = conf_scores[cl][c_mask]
                
                if scores.ndim == 0 or scores.numel() == 0:
                    continue
                    
                l_mask = c_mask.unsqueeze(1).expand(boxes.shape)
                boxes_ = boxes[l_mask].view(-1, 4)
                
                ids, count = nms(
                    boxes_, scores, self.nms_thresh, self.nms_top_k)
                count = count if count < self.top_k else self.top_k

                if count > 0:
                    selected_scores = scores[ids[:count]].unsqueeze(1)
                    selected_boxes = boxes_[ids[:count]]
                    output[i, cl, :count] = jt.concat((selected_scores, selected_boxes), dim=1)

        return output