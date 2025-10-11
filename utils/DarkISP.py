import jittor as jt
import numpy as np
import random
import scipy.stats as stats
from scipy.stats import truncnorm

def apply_ccm(image, ccm):
    '''
    The function of apply CCM matrix
    '''
    shape = image.shape
    image = image.view(-1, 3)
    image = jt.matmul(image, ccm)
    return image.view(shape)

def random_noise_levels():
    """Generates random shot and read noise from a log-log linear distribution."""
    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)
    log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = np.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + np.random.normal(scale=0.26)
    read_noise = np.exp(log_read_noise)
    return shot_noise, read_noise

def Low_Illumination_Degrading(img, safe_invert=False):
    '''
    (1)unprocess part(RGB2RAW) (2)low light corruption part (3)ISP part(RAW2RGB)
    Some code copy from 'https://github.com/timothybrooks/unprocessing', thx to their work ~
    input:
    img (Var): Input normal light images of shape (C, H, W).
    return:
    img_low (Var): Output degradation low light images of shape (C, H, W).
    para_gt (Var): Output degradation parameter in the whole process.
    '''

    '''
    parameter setting
    '''
    # 移除 device = img.device，Jittor 不需要这个
    config = dict(darkness_range=(0.01, 0.1),
                 gamma_range=(2.0, 3.5),
                 rgb_range=(0.8, 0.1),
                 red_range=(1.9, 2.4),
                 blue_range=(1.5, 1.9),
                 quantisation=[12, 14, 16])
    
    # camera color matrix
    xyz2cams = [[[1.0234, -0.2969, -0.2266],
                 [-0.5625, 1.6328, -0.0469],
                 [-0.0703, 0.2188, 0.6406]],
                [[0.4913, -0.0541, -0.0202],
                 [-0.613, 1.3513, 0.2906],
                 [-0.1564, 0.2151, 0.7183]],
                [[0.838, -0.263, -0.0639],
                 [-0.2887, 1.0725, 0.2496],
                 [-0.0627, 0.1427, 0.5438]],
                [[0.6596, -0.2079, -0.0562],
                 [-0.4782, 1.3016, 0.1933],
                 [-0.097, 0.1581, 0.5181]]]
    rgb2xyz = [[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]]

    '''
    (1)unprocess part(RGB2RAW): 1.inverse tone, 2.inverse gamma, 3.sRGB2cRGB, 4.inverse WB digital gains
    '''
    img1 = img.permute(1, 2, 0)  # (C, H, W) -- (H, W, C)
    
    # inverse tone mapping
    img1 = 0.5 - jt.sin(jt.asin(1.0 - 2.0 * img1) / 3.0)
    
    # inverse gamma
    epsilon = jt.array([1e-8], dtype=jt.float32)  # 移除 .to(device)
    gamma = random.uniform(config['gamma_range'][0], config['gamma_range'][1])
    img2 = jt.maximum(img1, epsilon) ** gamma
    
    # sRGB2cRGB
    xyz2cam = random.choice(xyz2cams)
    rgb2cam = np.matmul(xyz2cam, rgb2xyz)
    rgb2cam = jt.array(rgb2cam / np.sum(rgb2cam, axis=-1), dtype=jt.float32)  # 移除 .to(device)
    
    img3 = apply_ccm(img2, rgb2cam)

    # inverse WB
    rgb_gain = random.normalvariate(config['rgb_range'][0], config['rgb_range'][1])
    red_gain = random.uniform(config['red_range'][0], config['red_range'][1])
    blue_gain = random.uniform(config['blue_range'][0], config['blue_range'][1])

    gains1 = np.stack([1.0 / red_gain, 1.0, 1.0 / blue_gain]) * rgb_gain
    gains1 = gains1[np.newaxis, np.newaxis, :]
    gains1 = jt.array(gains1, dtype=jt.float32)  # 移除 .to(device)

    # color disorder !!!
    if safe_invert:
        img3_gray = jt.mean(img3, dim=-1, keepdims=True)
        inflection = 0.9
        zero = jt.zeros_like(img3_gray)  # 移除 .to(device)
        mask = (jt.maximum(img3_gray - inflection, zero) / (1.0 - inflection)) ** 2.0
        safe_gains = jt.maximum(mask + (1.0 - mask) * gains1, gains1)
        img4 = jt.clamp(img3 * safe_gains, min_v=0.0, max_v=1.0)
    else:
        img4 = img3 * gains1

    '''
    (2)low light corruption part: 5.darkness, 6.shot and read noise 
    '''
    # darkness(low photon numbers)
    lower, upper = config['darkness_range'][0], config['darkness_range'][1]
    mu, sigma = 0.1, 0.08
    darkness = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    darkness = darkness.rvs()
    
    img5 = img4 * darkness
    
    # add shot and read noise
    shot_noise, read_noise = random_noise_levels()
    var = img5 * shot_noise + read_noise  # here the read noise is independent
    var = jt.maximum(var, epsilon)
    
    # Jittor 正态分布生成方式不同
    noise = jt.normal(0, jt.sqrt(var))
    img6 = img5 + noise

    '''
    (3)ISP part(RAW2RGB): 7.quantisation  8.white balance 9.cRGB2sRGB 10.gamma correction
    '''
    # quantisation noise: uniform distribution
    bits = random.choice(config['quantisation'])
    # Jittor 的 uniform 函数参数略有不同
    quan_noise = (jt.random(img6.shape, dtype='float32', type='uniform') * 2 * (1 / (255 * bits))) - (1 / (255 * bits))
    img7 = img6 + quan_noise
    
    # white balance
    gains2 = np.stack([red_gain, 1.0, blue_gain])
    gains2 = gains2[np.newaxis, np.newaxis, :]
    gains2 = jt.array(gains2, dtype=jt.float32)  # 移除 .to(device)
    img8 = img7 * gains2

    # cRGB2sRGB
    cam2rgb = jt.linalg.inv(rgb2cam)
    img9 = apply_ccm(img8, cam2rgb)
    
    # gamma correction
    img10 = jt.maximum(img9, epsilon) ** (1 / gamma)

    img_low = img10.permute(2, 0, 1)  # (H, W, C) -- (C, H, W)

    # degradation information: darkness, gamma value, WB red, WB blue
    para_gt = jt.array([darkness, 1.0 / gamma, 1.0 / red_gain, 1.0 / blue_gain], dtype=jt.float32)  # 移除 .to(device)
    
    return img_low, para_gt