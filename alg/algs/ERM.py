# # coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.algs.base import Algorithm
from network.common_network import CNNnet
from main import CNN


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, args):
        super(ERM, self).__init__(args)
        self.cnn = CNN()
        self.args = args

    def update(self, minibatches, opt, sch):
        device = self.args.device
        loss = 0
        for m in range(len(minibatches)):
            x, y = minibatches[m][0].to(device).float(), minibatches[m][1].to(device).long()
            label = torch.squeeze(y - 1, dim=1).to(device)
            losses = F.cross_entropy(self.predict(x), label)
            loss = loss + losses
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'ERM': loss.item()}

    def predict(self, x):
        return self.cnn(x)
