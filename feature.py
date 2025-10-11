# -*- coding:utf-8 -*-
import cv2
import numpy as np
import jittor as jt
import jittor.nn as nn
import matplotlib.pyplot as plt
from models.factory import build_net

# 1️⃣ 启用 CUDA
jt.flags.use_cuda = 1

# 2️⃣ 构建并加载 DSFD 模型
net = build_net('test', num_classes=2)
net.eval()
net.load('./weights/dark/dsfd.pth')

# 3️⃣ 读取并预处理图片
img_path = "./10.png"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"未找到图片: {img_path}")

# 记录原始尺寸（方便之后恢复）
orig_h, orig_w = img.shape[:2]

# resize 到 640×640 并 BGR → RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (640, 640)).astype(np.float32)
img_resized -= np.array([104, 117, 123])
img_resized = img_resized.transpose(2, 0, 1)
img_tensor = jt.array(img_resized).unsqueeze(0)

# 4️⃣ 获取 backbone 特征
features = net.test_forward(img_tensor, return_features=True)
print(f"Feature 1 shape = {features[0].shape}")

# 5️⃣ 只取第一层特征（Feature 1）
feat = features[0]  # [B, C, H, W]

# 6️⃣ 对通道取平均，得到单通道特征图
feat_map = feat[0].mean(dim=0)  # [H, W]

# 7️⃣ 将特征图上采样回 **原图大小 (orig_h, orig_w)**
feat_resized = nn.interpolate(
    feat_map.unsqueeze(0).unsqueeze(0), 
    size=(orig_h, orig_w), 
    mode='bilinear'
)
feat_resized = feat_resized.squeeze().numpy()

# 归一化到 [0, 1]
feat_resized = (feat_resized - feat_resized.min()) / (feat_resized.max() - feat_resized.min() + 1e-6)

# 8️⃣ 可视化叠加
plt.figure(figsize=(8, 8))
plt.imshow(img_rgb)  # 原图
plt.imshow(feat_resized, cmap='plasma')  # 半透明叠加
plt.title(f"Feature 1 Upsampled to Original Size ({orig_w}×{orig_h})")
plt.axis("off")
plt.tight_layout()
plt.show()
