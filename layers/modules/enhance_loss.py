from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import jittor as jt
import jittor.nn as nn
from data.config import cfg

from typing import Union, Sequence, Optional, Tuple

def ssim(img1, img2, window_size=11, size_average=True):
    from jittor.nn import conv2d
    import math
    
    def create_window(window_size, channel):
        _1D_window = jt.array([math.exp(-(x - window_size//2)**2/float(2*1.5**2)) for x in range(window_size)])
        _2D_window = _1D_window.unsqueeze(1) * _1D_window.unsqueeze(0)
        _2D_window = _2D_window / jt.sum(_2D_window)
        _2D_window = _2D_window.unsqueeze(0).unsqueeze(0)
        return _2D_window.expand([channel, 1, window_size, window_size])
    
    (_, channel, height, width) = img1.shape
    window = create_window(window_size, channel)
    
    mu1 = conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01**2
    C2 = 0.03**2
    eps = 1e-8
    
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2 + eps))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gradient(input_tensor, direction):
    smooth_kernel_x = jt.array([[0, 0], [-1, 1]], dtype=jt.float32).view((1, 1, 2, 2))
    smooth_kernel_y = smooth_kernel_x.transpose(2, 3)

    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    grad_out = jt.abs(nn.conv2d(input_tensor, kernel,
                              stride=1, padding=1))
    return grad_out

def ave_gradient(input_tensor, direction):
    return nn.avg_pool2d(gradient(input_tensor, direction),
                        kernel_size=3, stride=1, padding=1)

def smooth(input_I, input_R):
    input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
    input_R = input_R.unsqueeze(dim=1)
    return jt.mean(gradient(input_I, "x") * jt.exp(-10 * ave_gradient(input_R, "x")) +
                  gradient(input_I, "y") * jt.exp(-10 * ave_gradient(input_R, "y")))

class EnhanceLoss(nn.Module):
    def __init__(self):
        super(EnhanceLoss, self).__init__()
        
    def execute(self, preds, img, img_dark):
        R_dark, R_light, R_dark_2, R_light_2, I_dark, I_light = preds

        losses_equal_R = (nn.mse_loss(R_dark, R_light.detach())) * cfg.WEIGHT.EQUAL_R
        losses_recon_low = nn.mse_loss(R_dark * I_dark, img_dark) * 1. + (1. - ssim(R_dark * I_dark, img_dark))
        losses_recon_high = nn.mse_loss(R_light * I_light, img) * 1. + (1. - ssim(R_light * I_light, img))

        losses_smooth_low = smooth(I_dark, R_dark) * cfg.WEIGHT.SMOOTH
        losses_smooth_high = smooth(I_light, R_light) * cfg.WEIGHT.SMOOTH
        # Redecomposition cohering loss
        losses_rc = (nn.mse_loss(R_dark_2, R_dark.detach()) + nn.mse_loss(R_light_2, R_light.detach())) * cfg.WEIGHT.RC

        enhance_loss = losses_equal_R + losses_recon_low + losses_recon_high + losses_smooth_low \
                       + losses_smooth_high + losses_rc

        return enhance_loss