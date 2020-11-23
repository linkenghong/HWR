import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import sys
from alfred.utils.log import logger as logging

class HWDB_AlexNet(nn.Module):
    def __init__(self, num_class):
        super(HWDB_AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, groups=2, padding=2)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, groups=2, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, groups=2, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(3*3*256, 4096)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(4096, num_class)

    def forward(self, x):
        x = self.norm1(F.relu(self.conv1(x)))
        x = self.pool1(x)
        x = self.norm2(F.relu(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        x = x.view(-1, 3 * 3 * 256)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop1(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        # dw
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class HWDB_MobileNet(nn.Module):
    def __init__(self, num_classes):
        super(HWDB_MobileNet, self).__init__()
        self.conv1 = conv_bn(3, 8, 1)  # 64x64x8
        self.conv2 = conv_bn(8, 16, 1)  # 64x64x16
        self.conv3 = conv_dw(16, 32, 1)  # 64x64x32
        self.conv4 = conv_dw(32, 32, 2)  # 32x32x32
        self.conv5 = conv_dw(32, 64, 1)  # 32x32x64
        self.conv6 = conv_dw(64, 64, 2)  # 16x16x64
        self.conv7 = conv_dw(64, 128, 1)  # 16x16x128
        self.conv8 = conv_dw(128, 128, 1)  # 16x16x128
        self.conv9 = conv_dw(128, 128, 1)  # 16x16x128
        self.conv10 = conv_dw(128, 128, 1)  # 16x16x128
        self.conv11 = conv_dw(128, 128, 1)  # 16x16x128
        self.conv12 = conv_dw(128, 256, 2)  # 8x8x256
        self.conv13 = conv_dw(256, 256, 1)  # 8x8x256
        self.conv14 = conv_dw(256, 256, 1)  # 8x8x256
        self.conv15 = conv_dw(256, 512, 2)  # 4x4x512
        self.conv16 = conv_dw(512, 512, 1)  # 4x4x512
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
        self.weight_init()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x9 = F.relu(x8 + x9)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x11 = F.relu(x10 + x11)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x14 = F.relu(x13 + x14)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x = x16.view(x16.size(0), -1)
        x = self.classifier(x)
        return x

    def weight_init(self):
        for layer in self.modules():
            self._layer_init(layer)

    def _layer_init(self, m):
        # 使用isinstance来判断m属于什么类型
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            # m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class Inception(nn.Module):
    def __init__(self, inD, x1D, x3D_1, x3D_2, x5D_1,  x5D_2, poolD):
        super(Inception, self).__init__()
        self.branch1x1 = nn.Conv2d(inD, x1D, kernel_size=1)
        self.branch3x3_1 = nn.Conv2d(inD, x3D_1, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(x3D_1, x3D_2, kernel_size=3, padding=1)
        self.branch5x5_1 = nn.Conv2d(inD, x5D_1, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(x5D_1, x5D_2, kernel_size=5, padding=2)
        self.branch_pool_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_2 = nn.Conv2d(inD, poolD, kernel_size=1)


    def forward(self, x):
        branch1x1 = F.relu(self.branch1x1(x))

        branch3x3 = F.relu(self.branch3x3_1(x))
        branch3x3 = F.relu(self.branch3x3_2(branch3x3))

        branch5x5 = F.relu(self.branch5x5_1(x))
        branch5x5 = F.relu(self.branch5x5_2(branch5x5))

        branch_pool = self.branch_pool_1(x)
        branch_pool = F.relu(self.branch_pool_2(branch_pool))

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, 1)

class HWDB_GoogLeNet(nn.Module):
    def __init__(self, num_class):
        super(HWDB_GoogLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        self.reduction1 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  ####
        self.inc1 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inc2 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  ####
        self.inc3 = Inception(480, 160, 112, 224, 24, 64, 64)
        self.inc4 = Inception(512, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.AvgPool2d(kernel_size=5, stride=3, padding=1)  ####
        self.reduction2 = nn.Conv2d(832, 128, kernel_size=1)
        self.fc1 = nn.Linear(2*2*128, 1024)
        self.drop1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(1024, num_class)
        # self.sm = nn.Softmax(dim=1)
        self.weight_init()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)
        x = F.relu(self.reduction1(x))
        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.pool2(x)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.pool3(x)
        x = self.inc3(x)
        x = self.inc4(x)
        x = self.pool4(x)
        x = F.relu(self.reduction2(x))
        x = x.view(-1, 2 * 2 * 128)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.fc2(x)
        # x = self.sm(x)
        return x

    def weight_init(self):
        for layer in self.modules():
            self._layer_init(layer)

    def _layer_init(self, m):
        # 使用isinstance来判断m属于什么类型
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class HWDB_SegNet(nn.Module):
    def __init__(self, num_class):
        super(HWDB_SegNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        self.reduction1 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  ####
        self.inc1 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inc2 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  ####
        self.inc3 = Inception(480, 160, 112, 224, 24, 64, 64)
        self.inc4 = Inception(512, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.AvgPool2d(kernel_size=5, stride=3, padding=1)  ####
        self.reduction2 = nn.Conv2d(832, 128, kernel_size=1)
        self.fc1 = nn.Linear(2*2*128, 128)
        self.drop1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(128, num_class)
        # self.sm = nn.Softmax(dim=1)
        self.weight_init()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)
        x = F.relu(self.reduction1(x))
        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.pool2(x)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.pool3(x)
        x = self.inc3(x)
        x = self.inc4(x)
        x = self.pool4(x)
        x = F.relu(self.reduction2(x))
        x = x.view(-1, 2 * 2 * 128)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.fc2(x)
        # x = self.sm(x)
        return x

    def weight_init(self):
        for layer in self.modules():
            self._layer_init(layer)

    def _layer_init(self, m):
        # 使用isinstance来判断m属于什么类型
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


if __name__ == "__main__":

    if len(sys.argv) <= 1:
        logging.error('send a pattern like this: {}'.format('G'))
    else:
        p = sys.argv[1]
        logging.info('show img from: {}'.format(p))
        if p == 'G':
            model = HWDB_GoogLeNet(6765).cuda()
            summary(model, input_size=(3, 120, 120), device='cuda')
        elif p == 'A':
            model = HWDB_AlexNet(6765).cuda()
            summary(model, input_size=(3, 108, 108), device='cuda')
        elif p == 'M':
            model = HWDB_MobileNet(6765).cuda()
            summary(model, input_size=(3, 64, 64), device='cuda')
        elif p == 'S':
            # torch.cuda.set_device(1)
            model = HWDB_SegNet(6765).cuda()
            summary(model, input_size=(3, 120, 120), device='cuda')