# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os

# 1️⃣ 设置路径
img_path = "./58.png"
txt_path = "./result/58.txt"
save_path = "./result/58_detected.png"

# 2️⃣ 读取图片
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"未找到图片: {img_path}")

# 3️⃣ 读取检测结果
if not os.path.exists(txt_path):
    raise FileNotFoundError(f"未找到检测结果文件: {txt_path}")

with open(txt_path, 'r') as f:
    lines = f.readlines()

# 4️⃣ 绘制检测框（不显示置信度）
for line in lines:
    vals = line.strip().split()
    if len(vals) != 5:
        continue
    xmin, ymin, xmax, ymax, score = map(float, vals)
    if score < 0.5:  # 可以根据需要过滤低置信度目标
        continue
    # 绘制矩形框（红色）
    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

# 5️⃣ 保存和显示结果
cv2.imwrite(save_path, img)
cv2.imshow("Detection Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"✅ 检测框已绘制并保存到: {save_path}")
