import os
from PIL import Image
import jittor as jt
import jittor.nn as nn

class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.channel = channel
        self.kernel_size = kernel_size
        
        # 计算各层需要的padding
        self.padding0 = (kernel_size * 3 - 1) // 2  # 第一层特殊处理
        self.padding_std = kernel_size // 2  # 标准卷积padding

        # 网络层定义（不需要padding参数）
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3, padding=0)
        
        self.net1_convs = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding=0),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=0),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=0),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=0),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=0),
            nn.ReLU()
        )
        
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size, padding=0)

    def execute(self, input_im):
        input_max = input_im.max(dim=1, keepdims=True)
        input_img = jt.concat((input_max, input_im), dim=1)
        
        # 第一层卷积 + replicate padding
        x = nn.pad(input_img, (self.padding0, self.padding0, self.padding0, self.padding0), mode='replicate')
        feats0 = self.net1_conv0(x)
        
        # 中间层
        x = feats0
        for layer in self.net1_convs:
            if isinstance(layer, nn.Conv2d):
                # 添加replicate padding
                x = nn.pad(x, (self.padding_std, self.padding_std, self.padding_std, self.padding_std), mode='replicate')
                x = layer(x)
            else:
                x = layer(x)
        
        # 最后一层
        x = nn.pad(x, (self.padding_std, self.padding_std, self.padding_std, self.padding_std), mode='replicate')
        outs = self.net1_recon(x)
        
        R = jt.sigmoid(outs[:, 0:3, :, :])
        L = jt.sigmoid(outs[:, 3:4, :, :])
        return R, L

# This Retinex Decom Net is frozen during training of DAI-Net
class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()

        self.DecomNet = DecomNet()

    def execute(self, input):
        # Forward DecompNet
        R, I = self.DecomNet(input)
        return R, I

    def loss(self, R_low, I_low, R_high, I_high, input_low, input_high):
        # Compute losses
        recon_loss_low = nn.l1_loss(R_low * I_low, input_low)
        recon_loss_high = nn.l1_loss(R_high * I_high, input_high)
        recon_loss_mutal_low = nn.l1_loss(R_high * I_low, input_low)
        recon_loss_mutal_high = nn.l1_loss(R_low * I_high, input_high)
        equal_R_loss = nn.l1_loss(R_low, R_high.detach())

        Ismooth_loss_low = self.smooth(I_low, R_low)
        Ismooth_loss_high = self.smooth(I_high, R_high)

        loss_Decom = recon_loss_low + \
                     recon_loss_high + \
                     0.001 * recon_loss_mutal_low + \
                     0.001 * recon_loss_mutal_high + \
                     0.1 * Ismooth_loss_low + \
                     0.1 * Ismooth_loss_high + \
                     0.01 * equal_R_loss
        return loss_Decom

    def gradient(self, input_tensor, direction):
        smooth_kernel_x = jt.array([[0, 0], [-1, 1]], dtype=jt.float32).view((1, 1, 2, 2))
        smooth_kernel_y = smooth_kernel_x.transpose(2, 3)
        

        if direction == "x":
            kernel = smooth_kernel_x
        elif direction == "y":
            kernel = smooth_kernel_y
        
        grad_out = jt.abs(nn.conv2d(input_tensor, kernel,
                                  stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return nn.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
        input_R = input_R.unsqueeze(dim=1)
        return jt.mean(self.gradient(input_I, "x") * jt.exp(-10 * self.ave_gradient(input_R, "x")) +
                       self.gradient(input_I, "y") * jt.exp(-10 * self.ave_gradient(input_R, "y")))