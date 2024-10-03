# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, args):
        super(GroupDRO, self).__init__(args)
        self.register_buffer("q", torch.Tensor())
        self.args = args

    def update(self, minibatches, opt, sch):
        device = self.args.device
        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m][0].cuda().float(), minibatches[m][1].cuda().long()
            data = torch.unsqueeze(x, 1).to(device)
            data = torch.cat((data, data), 2)
            label = torch.squeeze(y - 1, dim=1).to(device)
            losses[m] = F.cross_entropy(self.predict(data), label)
            self.q[m] *= (self.args.groupdro_eta * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()

        return {'group': loss.item()}
