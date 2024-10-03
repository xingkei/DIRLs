# 09-12
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from datautil import loadmat
import numpy as np
import torchmetrics
import torch.nn.functional as F
import itertools
from utils.fig import plot_fig, hunxiao
from scipy import io, signal
import sklearn


def our_args():
    parser = argparse.ArgumentParser(description='Working condition generalization')
    parser.add_argument('--num_classes', type=int, default=6, help='分类数')
    parser.add_argument('--batchSize', type=int, default=128, help='batch size')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle')
    parser.add_argument('--normalization', type=bool, default=False, help='normalization')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epoches', type=int, default=50, help='epoch')
    parser.add_argument('--bottleneck', type=int, default=256, help='bottleneck')
    parser.add_argument('--alpha', type=float, default=1, help='预测损失权重')
    parser.add_argument('--beta', type=float, default=0.0004, help='距离损失权重')
    parser.add_argument('--lam', type=float, default=1, help='coral损失权重')
    parser.add_argument('--distType', type=str, default='2-norm', help='距离损失权重')
    args = parser.parse_args()
    return args


# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Featurizer(nn.Module):
    def __init__(self):
        super(Featurizer, self).__init__()
        self.featurizer = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x):
        x = self.featurizer(x)
        return x.view(x.size(0), -1)


class Bottleneck(nn.Module):
    def __init__(self, fea_num):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Linear(128 * 318, fea_num)

    def forward(self, x):
        return self.bottleneck(x)


class Classifier(nn.Module):
    def __init__(self, fea_num, classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(fea_num, classes)

    def forward(self, x):
        return self.classifier(x)


# 域不变表示学习策略domain invariant representation learning strategy
class DIRLs(nn.Module):
    def __init__(self, args):
        super(DIRLs, self).__init__()
        self.args = args
        self.featurizer = Featurizer()
        self.bottleneck = Bottleneck(args.bottleneck)  # 全连接层1
        self.classifier = Classifier(args.bottleneck, args.num_classes)  # 全连接层2，输出类别数为6

        self.tfbd = args.bottleneck//2   # //表示地板除法

        self.teaf = Featurizer()
        self.teab = Bottleneck(self.tfbd)
        self.teac = Classifier(self.tfbd, args.num_classes)
        self.teaNet = nn.Sequential(
            self.teaf,
            self.teab,
            self.teac
        )

    def teanettrain(self, dataloaders, lr):
        teaModel = self.teaNet.to(device)
        optimizer = optim.Adam(teaModel.parameters(), lr)
        teaModel.train()
        minibatches_iterator = itertools.tee(dataloaders, self.args.epoches)
        for epoch in range(self.args.epoches):
            minibatches = [(tdata) for tdata in next(minibatches_iterator[epoch])]
            all_x = torch.cat([data[0].to(device).float() for data in minibatches])
            hilbert_envelope = signal.hilbert(all_x.cpu().numpy())
            all_z = torch.abs(torch.fft.fftn(torch.tensor(hilbert_envelope))).to(device)
            # all_z = torch.abs(torch.fft.fftn(all_x))
            all_y = torch.cat([data[1].to(device).long() for data in minibatches])
            optimizer.zero_grad()
            all_p = self.teaNet(torch.unsqueeze(all_z, dim=1))
            loss = F.cross_entropy(all_p, torch.squeeze(all_y-1), reduction='mean')
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print('epoch: %d, cls loss: %.4f' % (epoch, loss))
        teaModel.eval()

    def coral(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def update(self, minibatches, val_dataset, batchnum, epochs=30, lr=0.001):
        opt1 = optim.Adam(self.featurizer.parameters(), lr=lr)
        opt2 = optim.Adam(self.bottleneck.parameters(), lr=lr)
        opt3 = optim.Adam(self.classifier.parameters(), lr=lr)
        acc_list = []
        loss_list = []
        self.to(device)
        self.train()
        # 复制epoch个迭代器
        batchdata = itertools.tee(minibatches, epochs)
        for epoch in range(epochs):    # 迭代epoch
            total_loss = 0.0
            for batch in range(batchnum):    # 迭代batch
                opt1.zero_grad()
                opt2.zero_grad()
                opt3.zero_grad()
                minibatch = next(batchdata[epoch])
                all_x = torch.cat([data[0].cuda().float() for data in minibatch])
                all_y = torch.cat([data[1].cuda().long() for data in minibatch])
                with torch.no_grad():
                    all_x1 = torch.angle(torch.fft.fftn(all_x))
                    tfea = self.teab(self.teaf(torch.unsqueeze(all_x1, dim=1))).detach()

                all_z = self.bottleneck(self.featurizer(torch.unsqueeze(all_x, dim=1)))
                loss1 = F.cross_entropy(self.classifier(all_z), torch.squeeze(all_y-1))

                loss2 = F.mse_loss(all_z[:, :self.tfbd], tfea)*self.args.alpha
                if self.args.distType == '2-norm':
                    loss3 = -F.mse_loss(all_z[:, :self.tfbd],
                                        all_z[:, self.tfbd:])*self.args.beta
                elif self.args.distType == 'norm-2-norm':
                    loss3 = -F.mse_loss(all_z[:, :self.tfbd]/torch.norm(all_z[:, :self.tfbd], dim=1, keepdim=True),
                                        all_z[:, self.tfbd:]/torch.norm(all_z[:, self.tfbd:], dim=1, keepdim=True))*self.args.beta
                elif self.args.distType == 'norm-1-norm':
                    loss3 = -F.l1_loss(all_z[:, :self.tfbd]/torch.norm(all_z[:, :self.tfbd], dim=1, keepdim=True),
                                       all_z[:, self.tfbd:]/torch.norm(all_z[:, self.tfbd:], dim=1, keepdim=True))*self.args.beta
                elif self.args.distType == 'cos':
                    loss3 = torch.mean(F.cosine_similarity(
                        all_z[:, :self.tfbd], all_z[:, self.tfbd:]))*self.args.beta
                loss4 = 0
                if len(minibatch) > 1:
                    for i in range(len(minibatch)-1):
                        for j in range(i+1, len(minibatch)):
                            loss4 += self.coral(all_z[i*self.args.batchSize:(i+1)*self.args.batchSize, self.tfbd:],
                                                all_z[j*self.args.batchSize:(j+1)*self.args.batchSize, self.tfbd:])
                    loss4 = loss4*2/(len(minibatch)*(len(minibatch)-1))*self.args.lam
                else:
                    loss4 = self.coral(all_z[:self.args.batchSize//2, self.tfbd:],
                                       all_z[self.args.batchSize//2:, self.tfbd:])
                    loss4 = loss4*self.args.lam

                loss = loss1+loss2+loss3+loss4
                loss.backward()
                opt1.step()
                opt2.step()
                opt3.step()
                total_loss += loss.item()
            # self.eval()
            # val_acc = self.validation(val_dataset)
            # self.train()
            # print({'epoch': f'{epoch:.0f}', 'class': f'{loss1.item():.4f}', 'dist': f'{loss2.item():.4f}',
            #        'exp': f'{loss3.item():.4f}', 'align': f'{loss4.item():.4f}', 'total': f'{total_loss:.4f}',
            #        'val_acc': f'{val_acc:.4f}'})
            # acc_list.append(val_acc.item())
            loss_list.append(total_loss)
        self.eval()
        # plot_fig(range(1, epochs+1), loss_list, acc_list)
        # # 将list转换为numpy数组
        # my_array1 = np.array(loss_list)
        # my_array2 = np.array(acc_list)
        # # 使用scipy的savemat函数保存为.mat文件
        # io.savemat('./output/train_pad.mat', {'loss': my_array1, 'accuracy': my_array2})

    def predict(self, x):
        self.to(device)
        return self.classifier(self.bottleneck(self.featurizer(x)))

    def validation(self, val_data):
        metric = torchmetrics.Accuracy(task='multiclass', num_classes=6).to(device)
        with torch.no_grad():
            for inputs, labels in val_data:
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到 GPU 上
                outputs = self.predict(torch.unsqueeze(inputs, dim=1))
                _, predicted = torch.max(outputs, 1)
                batch_acc = metric(predicted, (labels - 1).squeeze())
        total_acc = metric.compute()
        return total_acc


# 针对paderborn数据集
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # 计算全连接层输入特征数量，需要根据卷积层的输出维度和输入样本的特征维度计算
        self.bottleneck = nn.Linear(128 * 318, 256)  # 全连接层1
        self.classifier = nn.Linear(256, 6)  # 全连接层2，输出类别数为6

    def forward(self, x):
        x = x.unsqueeze(1)  # 将输入张量增加一个维度以适应卷积层的输入要求
        # 输入x的维度应该为 (batch_size, 1, 2560)
        x = self.feature(x)

        # 将卷积层的输出展平
        x = x.view(x.size(0), -1)

        x = F.relu(self.bottleneck(x))
        x = self.classifier(x)
        return x


# 创建数据集和数据加载器
def create_dataset(path, Normalize):
    # 创建数据集和数据加载器
    dataset = loadmat.TrainDataset(path, Normalize)
    return dataset


# 训练模型
def train_model(model, minibatches, batchnum, epochs=30, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)  # 将模型移动到 GPU 上
    model.train()
    # 复制epoches个迭代器供每个训练epoch使用
    batchdata = itertools.tee(minibatches, epochs)

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in range(batchnum):
            # 从迭代器中获取源域s样本并cat
            minibatch = next(batchdata[epoch])
            inputs = torch.cat([data[0].float() for data in minibatch])
            labels = torch.cat([data[1].long() for data in minibatch])
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到 GPU
            outputs = model(inputs)  # 添加一个维度以适应卷积层的输入要求
            label = torch.squeeze(labels - 1, dim=1)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print("Epoch %d, Loss:%.2f" % (epoch + 1, running_loss))


# 测试模型
def test_model(model, dataloader):
    metric = torchmetrics.Accuracy(task='multiclass', num_classes=6).to(device)
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到 GPU 上
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            batch_acc = metric(predicted, (labels - 1).squeeze())
    total_acc = metric.compute()
    print("Accuracy: %.3f %% of ERM" % (total_acc * 100))


class TestDIRLs:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def acc(self):
        metric = torchmetrics.Accuracy(task='multiclass', num_classes=6).to(device)
        with torch.no_grad():
            for inputs, labels in self.data:
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到 GPU 上
                outputs = self.model.predict(torch.unsqueeze(inputs, dim=1))
                _, predicted = torch.max(outputs, 1)
                batch_acc = metric(predicted, (labels - 1).squeeze())
        total_acc = metric.compute()
        print("Accuracy: %.3f %% of the DIFEX" % (total_acc * 100))
        # hunxiao(predicted, (labels - 1).squeeze(), 6)
        # 将list转换为numpy数组
        my_array3 = np.array((predicted+1).cpu().tolist())
        my_array4 = np.array(labels.squeeze().cpu().tolist())
        # 使用scipy的savemat函数保存为.mat文件
        io.savemat('./output/con_pad.mat', {'y_pred': my_array3, 'y_true': my_array4})
        return total_acc


def set_random_seed(seed=0):
    # seed setting
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    # 固定随机种子
    set_random_seed()
    # 导入参数
    args = our_args()
    # 参数分析
    learning_rates = np.arange(0.005, 0.105, 0.005)
    batch_sizes = np.array([2 ** i for i in range(2, 10)])
    accuracy_map = np.zeros((len(learning_rates), len(batch_sizes)))
    for i, lr in enumerate(learning_rates):
        for j, bs in enumerate(batch_sizes):
            print('批大小：%d' %bs)
            # 创建数据加载器
            dataset1 = create_dataset('./paderborn/condition1.mat', Normalize=args.normalization)
            trainloader1 = DataLoader(dataset1, batch_size=int(bs), shuffle=args.shuffle, drop_last=True)
            dataset2 = create_dataset('./paderborn/condition2.mat', Normalize=args.normalization)
            trainloader2 = DataLoader(dataset2, batch_size=int(bs), shuffle=args.shuffle, drop_last=True)
            dataset3 = create_dataset('./paderborn/condition3.mat', Normalize=args.normalization)
            trainloader3 = DataLoader(dataset3, batch_size=int(bs), shuffle=args.shuffle, drop_last=True)
            dataset4 = create_dataset('./paderborn/condition4.mat', Normalize=args.normalization)

            # trainset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3])
            # trainlaoder = DataLoader(trainset, batch_size=args.batchSize, shuffle=False, drop_last=True)

            testloader = DataLoader(dataset4, batch_size=int(bs), shuffle=False, drop_last=True)
            # 初始化模型
            model = ConvNet()
            model1 = DIRLs(args)
            # 融合多个dataset
            batchnum = len(trainloader1)
            minibatch = itertools.tee(zip(trainloader1, trainloader2, trainloader3), 3)
            # 训练教师网络
            MyDirls = DIRLs(args=args)
            MyDirls.teanettrain(dataloaders=minibatch[0], lr=lr)
            print('教师模型训练完毕...\n')

            # # 训练原始模型
            # train_model(model, minibatch[2], batchnum=batchnum)
            # # 测试原始模型
            # test_model(model, testloader)

            # 训练DIRLs模型
            model1.update(minibatches=minibatch[1], val_dataset=testloader, batchnum=batchnum, lr=lr)
            # 测试DIRLs
            AccDIRLs = TestDIRLs(model1, testloader)
            acc = AccDIRLs.acc()
            accuracy_map[i, j] = acc

    # 设置字体和字号
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    # 绘制热力图
    LR, BS = np.meshgrid(learning_rates, batch_sizes)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(LR, BS, accuracy_map.T, cmap='viridis')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Batch Size')
    ax.set_zlabel('Accuracy')
    plt.show()
    # 使用scipy的savemat函数保存为.mat文件
    io.savemat('./output/relitu.mat', {'learningGrid': LR, 'batchSizeGrid': BS, 'accuracyGrid': accuracy_map.T})

