from __future__ import absolute_import, division, print_function
import jittor as jt
from jittor.dataset import Dataset
from PIL import Image
import numpy as np
import random
from utils.augmentations import preprocess


class WIDERDetection(Dataset):
    """WIDER Face 数据集的 Jittor 实现"""
    def __init__(self, list_file, mode='train', sample_ratio=1.0,
                 batch_size=4, shuffle=True, drop_last=True):
        super(WIDERDetection, self).__init__()
        self.mode = mode
        self.fnames = []
        self.boxes = []
        self.labels = []

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            num_faces = int(line[1])
            box, label = [], []
            for i in range(num_faces):
                x = float(line[2 + 5 * i])
                y = float(line[3 + 5 * i])
                w = float(line[4 + 5 * i])
                h = float(line[5 + 5 * i])
                c = int(line[6 + 5 * i])
                if w <= 0 or h <= 0:
                    continue
                box.append([x, y, x + w, y + h])
                label.append(c)
            if len(box) > 0:
                self.fnames.append(line[0])
                self.boxes.append(box)
                self.labels.append(label)

        self.num_samples = len(self.boxes)

        if sample_ratio < 1.0:
            keep_samples = int(self.num_samples * sample_ratio)
            keep_indices = random.sample(range(self.num_samples), keep_samples)
            self.fnames = [self.fnames[i] for i in keep_indices]
            self.boxes = [self.boxes[i] for i in keep_indices]
            self.labels = [self.labels[i] for i in keep_indices]
            self.num_samples = len(self.fnames)
            print(f"训练集随机采样 {sample_ratio*100:.1f}% 样本，保留 {self.num_samples} 个")

        self.set_attrs(
            total_len=self.num_samples,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )

    def __getitem__(self, index):
        img, target, img_path, h, w = self.pull_item(index)
        return img, target, img_path

    def pull_item(self, index):
        while True:
            image_path = self.fnames[index]
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')

            im_width, im_height = img.size
            boxes = self.annotransform(np.array(self.boxes[index]), im_width, im_height)
            label = np.array(self.labels[index])
            bbox_labels = np.hstack((label[:, np.newaxis], boxes)).tolist()

            img, sample_labels = preprocess(img, bbox_labels, self.mode, image_path)
            sample_labels = np.array(sample_labels)

            if len(sample_labels) > 0:
                target = np.hstack(
                    (sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))
                assert (target[:, 2] > target[:, 0]).any()
                assert (target[:, 3] > target[:, 1]).any()
                break
            else:
                index = random.randrange(0, self.num_samples)

        img = np.array(img)
        img = jt.array(img)
        return img, jt.array(target), image_path, im_height, im_width

    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes

    def collate_batch(self, batch):
        """处理批次数据，适应不同数量的标注框"""
        images, targets, paths = [], [], []
        for img, target, path in batch:
            images.append(img)
            targets.append(target)
            paths.append(path)
        images = jt.stack(images, dim=0)
        return images, targets, paths


if __name__ == '__main__':
    from config import cfg
    dataset = WIDERDetection(
        cfg.FACE.TRAIN_FILE, 
        mode='train', 
        sample_ratio=0.5, 
        batch_size=1, 
        shuffle=True, 
        drop_last=True
    )