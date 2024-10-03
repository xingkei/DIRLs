import argparse
import torch
from torch import nn
from datetime import datetime
import torch.optim as optim
from datautil import loadmat
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import random
import numpy as np
import pandas as pd

USE_CUDA = torch.cuda.is_available()


def cnn_args():
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--source', type=str, default='..\GearData\Condition123_4\Con123.mat')
    parser.add_argument('--target', type=str, default='..\GearData\Condition3\Con3.mat')
    args = parser.parse_args()
    return args


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
            # nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 16, 3, 1),
            # nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 1),

            nn.Conv1d(16, 16, 3, 1),
            # nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 16, kernel_size=3, stride=1),
            # nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        self.linea_layer = nn.Sequential(
            nn.Linear(16 * 23, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn_layer(x)
        x = x.view(-1, 16 * 23)
        x = self.linea_layer(x)

        return x


if __name__ == '__main__':
    args = cnn_args()
    device = args.device
    set_random_seed(0)
    # model = ResNet(ResidualBlock).to(device)
    model = CNNnet().to(device)
    writer = SummaryWriter()
    batchsize = 1024
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    traindata = loadmat.TrainDataset(args.source)
    dataloader = DataLoader(dataset=traindata, batch_size=batchsize, pin_memory=True, shuffle=True, drop_last=False)
    testdata = loadmat.TrainDataset(args.target)
    testdataloader = DataLoader(dataset=testdata, batch_size=batchsize, pin_memory=True, shuffle=False, drop_last=False)
    final_acc = 0
    filename = datetime.now().strftime('%d-%m-%y-%H_%M.pth')
    for i in range(2000):
        model.train()
        total_loss = 0
        for i_batch, (batch_data, batch_label) in enumerate(dataloader):
            data = torch.unsqueeze(batch_data, 1).to(device)
            data = torch.cat((data, data), 2)
            label = torch.squeeze(batch_label - 1, dim=1).to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, label)
            total_loss = total_loss + loss.item()
            loss.backward()
            optimizer.step()

        if i % 30 == 0:
            model.eval()
            pre_total = []
            with torch.no_grad():
                total_num = 0
                for testbatch, (testX, testY) in enumerate(testdataloader):
                    testX = torch.unsqueeze(testX, dim=1).to(device)
                    testX = torch.cat((testX, testX), 2)
                    testY = torch.squeeze(testY - 1, dim=1).to(device)
                    predictY = torch.argmax(model(testX), -1)
                    num = testY.eq(predictY).sum().item()
                    total_num = total_num + num
                    pre_total = np.append(pre_total, predictY.cpu().numpy())
                acc = total_num / testdata.__len__()
                print('test accuracy: %f' % acc)
                writer.add_scalar('test accuracy:', acc, i)
                if final_acc < acc:
                    final_acc = acc
                    torch.save(model, '../model/' + filename)
                    save = pd.DataFrame(pre_total, columns=['predict label'])
                    save.to_csv('../model/predict_label.csv')
        print('epoch: %d, loss: %f' % (i, total_loss))
        writer.add_scalar('training loss:', total_loss, i)
    writer.close()
