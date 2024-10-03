# coding=utf-8
import torch
from network import img_network
import numpy as np


def get_fea(args):
    if args.dataset == 'dg5':
        net = img_network.DTNBase()
    elif args.net.startswith('res'):
        net = img_network.ResBase(args.net)
    else:
        net = img_network.VGGBase(args.net)
    return net


def accuracy(network, loader, args):
    correct = 0
    total = 0
    pre_total=[]
    device = args.device
    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].to(device).float()
            y = data[1].to(device).long()
            p = network.predict(x)

            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
            pre_total = np.append(pre_total, p.argmax(1).cpu().numpy())
    network.train()
    return correct / total,  pre_total
