# coding=utf-8
import torch
import torch.nn as nn
from torch import autograd
import torch
from network.util import init_weights
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    子 module: Residual Block ---- ResNet 中一个跨层直连的单元
    """

    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(outchannel)
        )
        self.shortcut = nn.Sequential()
        # 如果输入和输出的通道不一致，或其步长不为 1，需要将二者转成一致
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)  # 输出 + 输入
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    实现主 module: ResNet-18
    ResNet 包含多个 layer, 每个 layer 又包含多个 residual block (上面实现的类)
    因此, 用 ResidualBlock 实现 Residual 部分，用 _make_layer 函数实现 layer
    """

    def __init__(self, ResidualBlock=ResidualBlock):
        super(ResNet, self).__init__()
        self.inchannel = 64
        # 最开始的操作
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        # 四个 layer， 对应 2， 3， 4， 5 层， 每层有两个 residual block
        self.layer1 = self._make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, 2, stride=2)
        # 最后的全连接，分类时使用
        self.fc = nn.Linear(128, 320)

    def _make_layer(self, block, channels, num_blocks, stride):
        """
        构建 layer, 每一个 layer 由多个 residual block 组成
        在 ResNet 中，每一个 layer 中只有两个 residual block
        """
        layers = []
        for i in range(num_blocks):
            if i == 0:  # 第一个是输入的 stride
                layers.append(block(self.inchannel, channels, stride))
            else:  # 后面的所有 stride，都置为 1
                layers.append(block(channels, channels, 1))
            self.inchannel = channels
        return nn.Sequential(*layers)  # 时序容器。Modules 会以他们传入的顺序被添加到容器中。

    def forward(self, x):
        # 最开始的处理
        out = self.conv1(x)
        # 四层 layer
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 全连接 输出分类信息
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        return self.fc(out)

    def feature_size(self):
        # Calculate the output size of the CNN
        return self.layer4(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)


class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 16, 3, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 1),

            nn.Conv1d(16, 16, 3, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 16, kernel_size=3, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        self.linea_layer = nn.Sequential(
            nn.Linear(16*23, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn_layer(x)
        x = x.view(-1, 16*23)
        x = self.linea_layer(x)
        return x

    def feature_shape(self):
        return self.cnn_layer(torch.zeros(1, 1, 16)).view(1, -1).size(1)

    def feature_size(self):
        return self.cnn_layer(torch.zeros(1, 1, 16).cuda()).view(1, -1).size(1)


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)  # 权值参数初始化
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return self.relu(x)


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(
                nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        return x


class feat_classifier_two(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier_two, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x
