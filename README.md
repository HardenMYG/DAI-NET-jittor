# DAI-NET-jittor
Jittor Implementation of the DAI-NET Model

# 环境配置

### 最终配置

WSL Ubuntu 22.04.5 LTS

Python 3.8.20

jittor 1.3.10.0

cuda 11.5

### 配置过程



# 数据准备脚本
```python
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
```

# 训练脚本

```python
save_folder = os.path.join(args.save_folder, 'dark')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

train_dataset = WIDERDetection(cfg.FACE.TRAIN_FILE, mode='train', batch_size=1, 
        shuffle=True, drop_last=True,sample_ratio=1.0)

val_dataset = WIDERDetection(cfg.FACE.VAL_FILE, mode='val',batch_size=1, 
        shuffle=False, drop_last=True, sample_ratio=1.0)


train_loader = DataLoader(train_dataset)

val_loader = DataLoader(val_dataset)


min_loss = np.inf

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


def train():
    per_epoch_size = len(train_dataset) // args.batch_size
    start_epoch = 0
    iteration = 0
    step_index = 0

    csv_file = os.path.join(save_folder, 'training_log.csv')

    basenet = basenet_factory()
    dsfd_net = build_net('train', cfg.NUM_CLASSES)
    net = dsfd_net
    net_enh = RetinexNet()
    net_enh.load_state_dict(jt.load(args.save_folder + 'decomp.pth'))

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        start_epoch = net.load_weights(args.resume)+1
        iteration = start_epoch * per_epoch_size
    else:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'iteration', 'total_loss'
            ])
        pth_file = args.save_folder + basenet
        load_vgg_pth_to_jt(net.vgg, pth_file) 



    if not args.resume:
        print('Initializing weights...')
        net.extras.apply(net.weights_init)
        net.fpn_topdown.apply(net.weights_init)
        net.fpn_latlayer.apply(net.weights_init)
        net.fpn_fem.apply(net.weights_init)
        net.loc_pal1.apply(net.weights_init)
        net.conf_pal1.apply(net.weights_init)
        net.loc_pal2.apply(net.weights_init)
        net.conf_pal2.apply(net.weights_init)
        net.ref.apply(net.weights_init)

    # Scaling the lr
    lr = args.lr * np.round(np.sqrt(args.batch_size / 4), 4)
    param_group = []
    param_group += [{'params': dsfd_net.vgg.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.extras.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_topdown.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_latlayer.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_fem.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.loc_pal1.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.conf_pal1.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.loc_pal2.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.conf_pal2.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.ref.parameters(), 'lr': lr / 10.}]

    optimizer = optim.SGD(param_group, lr=lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)


    criterion = MultiBoxLoss(cfg)
    criterion_enhance = EnhanceLoss()
    print('Loading wider dataset...')
    # print(f"验证集标注文件路径：{cfg.FACE.VAL_FILE}")  # 查看路径是否正确
    # print(f"验证集标注文件是否存在：{os.path.exists(cfg.FACE.VAL_FILE)}")  # 确认文件存在
    # print(f"验证集采样后样本数：{len(val_dataset)}")  # 关键！若为0，说明解析失败
    # print(f"val_loader 批次数量：{len(val_loader)}") 
    print('Using the specified args:')
    print(args)

    for step in cfg.LR_STEPS:
        if iteration > step:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
    
    net_enh.eval()
    net.train()
    
    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0

        for batch_idx, (images, targets, _) in enumerate(train_loader):
            images = images / 255.
            targetss = [ann for ann in targets]
            
            img_dark = jt.zeros((images.shape[0], images.shape[1], images.shape[2], images.shape[3]))
            
            # Generation of degraded data and AET groundtruth
            for i in range(images.shape[0]):
                img_dark_i, _ = Low_Illumination_Degrading(images[i])
                img_dark[i] = img_dark_i

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            t0 = time.time()
            with jt.no_grad():
                R_dark_gt, I_dark = net_enh(img_dark)
                R_light_gt, I_light = net_enh(images)

            out, out2, loss_mutual = net(img_dark, images, I_dark, I_light)
            R_dark, R_light, R_dark_2, R_light_2 = out2

            # backprop
            optimizer.zero_grad()

            loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targetss)
            loss_l_pa12, loss_c_pal2 = criterion(out[3:], targetss)

            loss_enhance = criterion_enhance([R_dark, R_light, R_dark_2, R_light_2, I_dark, I_light], images, img_dark) * 0.1
            loss_enhance2 = nn.l1_loss(R_dark, R_dark_gt) + nn.l1_loss(R_light, R_light_gt) + (
                        1. - ssim(R_dark, R_dark_gt)) + (1. - ssim(R_light, R_light_gt))  #ref

            loss = loss_l_pa1l + loss_c_pal1 + loss_l_pa12 + loss_c_pal2 + loss_enhance2 + loss_enhance + loss_mutual #mfa
            optimizer.backward(loss)
            optimizer.clip_grad_norm( max_norm=35, norm_type=2)
            optimizer.step()
            t1 = time.time()
            losses += loss.item()

            if iteration % 100 == 0:
                tloss = losses / (batch_idx + 1)
                print('Timer: %.4f' % (t1 - t0))
                print('epoch:' + repr(epoch) + ' || iter:' +
                      repr(iteration) + ' || Loss:%.4f' % (tloss))
                print('->> pal1 conf loss:{:.4f} || pal1 loc loss:{:.4f}'.format(
                    loss_c_pal1.item(), loss_l_pa1l.item()))
                print('->> pal2 conf loss:{:.4f} || pal2 loc loss:{:.4f}'.format(
                    loss_c_pal2.item(), loss_l_pa12.item()))
                print('->> enhance loss:{:.4f}'.format(loss_enhance.item()))
                print('->> enhance2 loss:{:.4f}'.format(loss_enhance2.item()))
                print('->> mutual loss:{:.4f}'.format(loss_mutual.item()))
                print('->>lr:{}'.format(optimizer.param_groups[0]['lr']))

                with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, iteration, tloss
                    ])

            if iteration != 0 and iteration % 5000 == 0:
                print('Saving state, iter:', iteration)
                file = 'dsfd_' + repr(iteration) + '.pth'
                jt.save(dsfd_net.state_dict(),
                           os.path.join(save_folder, file))
            iteration += 1
        
        if (epoch + 1) >= 0:
            val(epoch, net, dsfd_net, net_enh, criterion)
        if iteration >= cfg.MAX_STEPS:
            break


def val(epoch, net, dsfd_net, net_enh, criterion):
    net.eval()
    step = 0
    losses = 0.
    t1 = time.time()

    for batch_idx, (images, targets, img_paths) in enumerate(val_loader):
        images = images / 255.
            
        img_dark = jt.stack([Low_Illumination_Degrading(images[i])[0] for i in range(images.shape[0])])
        out, R = net.test_forward(img_dark)

        loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
        loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)
        loss = loss_l_pa12 + loss_c_pal2

        losses += loss.item()
        step += 1

    tloss = losses / step
    t2 = time.time()
    print('Timer: %.4f' % (t2 - t1))
    print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    if tloss < min_loss:
        print('Saving best state,epoch', epoch)
        jt.save(dsfd_net.state_dict(), os.path.join(
            save_folder, 'dsfd.pth'))
        min_loss = tloss

    states = {
        'epoch': epoch,
        'weight': dsfd_net.state_dict(),
    }
    jt.save(states, os.path.join(save_folder, 'dsfd_checkpoint.pth'))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma
```

# 测试脚本

```python
def tensor_to_image(tensor):
    grid = make_grid(tensor)
    ndarr = grid.multiply(255).add(0.5).clamp(0, 255).permute(1, 2, 0).uint8().numpy()
    return ndarr

def to_chw_bgr(image):
    """
    Transpose image from HWC to CHW and from RBG to BGR.
    Args:
        image (np.array): an image with HWC and RBG layout.
    """
    # HWC to CHW
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # RBG to BGR
    image = image[[2, 1, 0], :, :]
    return image

def detect_face(img, tmp_shrink):
    image = cv2.resize(img, None, None, fx=tmp_shrink,
                       fy=tmp_shrink, interpolation=cv2.INTER_LINEAR)

    x = to_chw_bgr(image)
    x = x.astype('float32')
    x = x / 255.
    x = x[[2, 1, 0], :, :]

    x = jt.array(x).unsqueeze(0)

    y = net.test_forward(x)[0]
    detections = y.data
    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

    boxes=[]
    scores = []
    for i in range(detections.shape[1]):
      j = 0
      while ((j < detections.shape[2]) and detections[0, i, j, 0] > 0.0):
        pt = (detections[0, i, j, 1:] * scale)
        score = detections[0, i, j, 0]
        boxes.append([pt[0],pt[1],pt[2],pt[3]])
        scores.append(score)
        j += 1

    det_conf = np.array(scores)
    boxes = np.array(boxes)

    if boxes.shape[0] == 0:
        return np.array([[0,0,0,0,0.001]])

    det_xmin = boxes[:,0]
    det_ymin = boxes[:,1]
    det_xmax = boxes[:,2]
    det_ymax = boxes[:,3]
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    return det


def flip_test(image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def multi_scale_test(image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(image, st)
    if max_im_shrink > 0.75:
        det_s = np.row_stack((det_s,detect_face(image, 0.75)))
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(image, bt)

    # enlarge small iamge x times for small face
    if max_im_shrink > 1.5:
        det_b = np.row_stack((det_b,detect_face(image, 1.5)))
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, detect_face(image, bt)))
            bt *= 2

        det_b = np.row_stack((det_b, detect_face(image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b


def multi_scale_test_pyramid(image, max_shrink):
    det_b = detect_face(image, 0.25)
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
        > 30)[0]
    det_b = det_b[index, :]

    st = [1.25, 1.75, 2.25]
    for i in range(len(st)):
        if (st[i] <= max_shrink):
            det_temp = detect_face(image, st[i])
            # enlarge only detect small face
            if st[i] > 1:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                det_temp = det_temp[index, :]
            else:
                index = np.where(
                    np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))
    return det_b


def bbox_vote(det_):
    order_ = det_[:, 4].ravel().argsort()[::-1]
    det_ = det_[order_, :]
    dets_ = np.zeros((0, 5),dtype=np.float32)
    while det_.shape[0] > 0:
        # IOU
        area_ = (det_[:, 2] - det_[:, 0] + 1) * (det_[:, 3] - det_[:, 1] + 1)
        xx1 = np.maximum(det_[0, 0], det_[:, 0])
        yy1 = np.maximum(det_[0, 1], det_[:, 1])
        xx2 = np.minimum(det_[0, 2], det_[:, 2])
        yy2 = np.minimum(det_[0, 3], det_[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o_ = inter / (area_[0] + area_[:] - inter)

        # get needed merge det and delete these det
        merge_index_ = np.where(o_ >= 0.7)[0]
        det_accu_ = det_[merge_index_, :]
        det_ = np.delete(det_, merge_index_, 0)

        if merge_index_.shape[0] <= 1:
            continue
        det_accu_[:, 0:4] = det_accu_[:, 0:4] * np.tile(det_accu_[:, -1:], (1, 4))
        max_score_ = np.max(det_accu_[:, 4])
        det_accu_sum_ = np.zeros((1, 5))
        det_accu_sum_[:, 0:4] = np.sum(det_accu_[:, 0:4], axis=0) / np.sum(det_accu_[:, -1:])
        det_accu_sum_[:, 4] = max_score_
        try:
            dets_ = np.row_stack((dets_, det_accu_sum_))
        except:
            dets_ = det_accu_sum_

    dets_ = dets_[0:750, :]
    return dets_


def load_models():
    print('build network')
    net = build_net('test', num_classes=2)
    net.eval()
    net.load('weights/dark/dsfd.pth') # Set the dir of your model weight

    return net
```

# 训练日志

### jittor

#### 训练Loss曲线
<img width="3600" height="1500" alt="image" src="https://github.com/user-attachments/assets/d41c8660-1048-4081-99a5-227ec3497ea4" />

#### GPU使用情况

<img width="1850" height="1365" alt="Screenshot 2025-10-07 155654" src="https://github.com/user-attachments/assets/880b7f11-9fae-4a47-ab82-ff68d5eac7e6" />


### pytorch

#### 训练Loss曲线

进行了测试，但暂未进行结果的记录，因为单个iteration需要6+s进行训练。对于我的设置来说batch_size为1，train数据集大小为1w+，一个epoch需要1w+的iteration，17个小时。

#### GPU使用情况

<img width="1844" height="1273" alt="Screenshot 2025-10-11 203728" src="https://github.com/user-attachments/assets/b769fa4a-c96d-47a7-b697-eb17f47bf8cd" />


# 模型效果

### 原图
<img width="1080" height="720" alt="image" src="https://github.com/user-attachments/assets/b9b4dce1-64a3-47d5-95f1-8d4ff7a75626" />

### 检测结果
<img width="1080" height="720" alt="image" src="https://github.com/user-attachments/assets/a00b2e65-5ce4-41da-9194-f593ba5885df" />

### 特征图
<img width="2130" height="1422" alt="Screenshot 2025-10-08 133732" src="https://github.com/user-attachments/assets/c68cc0e9-81ef-431b-aff2-b13ecb019594" />


# 其他说明

### 代码参考
本代码参照论文[《Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation》](https://zpdu.github.io/DAINet_page/)及其[代码](https://github.com/ZPDu/DAI-Net)进行jittor版本代码实现。

### 新添文件
'visual.py'用于生成目标检测框，'feature.py'用于生成模型中的特征图。
