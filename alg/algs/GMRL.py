# coding=utf-8
import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from network.common_network import CNNnet, feat_bottleneck, feat_classifier, ResNet
from alg.algs.base import Algorithm


class GMRL(Algorithm):    # 继承ERM
    def __init__(self, args):
        super(GMRL, self).__init__(args)
        rsc_f_drop_factor = 0.2
        rsc_b_drop_factor = 0.3
        self.drop_f = (1 - rsc_f_drop_factor) * 100
        self.drop_b = (1 - rsc_b_drop_factor) * 100
        self.num_classes = args.num_classes
        self.cnn = ResNet()
        self.bottleneck = feat_bottleneck(
            320, args.bottleneck, args.layer)    # 一层线性全连接  32表示卷积输出维度
        self.classifier = feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)          # 分类层，对应分类类别

    def update(self, x, y, opt, sch):
        all_x = torch.unsqueeze(x, 1).cuda()  # 获取训练数据及标签  data = torch.unsqueeze(x, 1).to(device)
        all_x = torch.cat((all_x, all_x), 2)
        all_y = y-1
        all_y = all_y.cuda()
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)    # 标签转化为独热编码
        all_f = self.bottleneck(self.cnn(all_x))
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.cuda()).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()

        return loss.item()

    def predict(self, x):
        return self.classifier(self.bottleneck(self.cnn(x)))
