import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, auc, roc_curve


def plot_fig(epochs, loss, accuracy):
    # 设置字体和字体大小
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16

    # 创建画布
    fig, ax1 = plt.subplots()

    # 绘制左侧坐标轴
    color = 'tab:red'
    ax1.set_xlabel('Epoch', fontname='Times New Roman', fontsize=16)
    ax1.set_ylabel('Loss', color=color, fontname='Times New Roman', fontsize=16)
    ax1.plot(epochs, loss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建右侧坐标轴
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color, fontname='Times New Roman', fontsize=16)
    ax2.plot(epochs, accuracy, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # 设置右侧坐标轴刻度从0开始
    ax2.set_ylim(0.2, 1)

    # 添加标题
    plt.title('Loss and Accuracy Over Epochs', fontname='Times New Roman', fontsize=16)

    # 展示图形
    plt.show()


def hunxiao(y_true, y_pred, num_class):

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    classes = range(1, num_class+1)

    # 设置字体和字体大小
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16

    # 绘制混淆矩阵图
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # 在图中显示混淆矩阵的值
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontname='Times New Roman', fontsize=16)

    plt.tight_layout()
    plt.ylabel('True label', fontname='Times New Roman', fontsize=16)
    plt.xlabel('Predicted label', fontname='Times New Roman', fontsize=16)
    plt.show()


def ROCs(y_pred_probs, y_true_onehot, num_class):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 26
    # y_pred_probs = model(testX).cpu().numpy()
    # y_true_onehot = torch.nn.functional.one_hot(testY.cpu(), num_classes=6)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算所有类别的平均AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_class

    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))

    plt.plot(all_fpr, mean_tpr, color='b',
             label='Mean ROC (area = {0:0.2f})'
                   ''.format(np.mean(list(roc_auc.values()))),
             lw=2)

    for i in range(num_class):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i+1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC)')
    leg = plt.legend(loc="lower right", fontsize=26)
    for text in leg.get_texts():
        text.set_alpha(0.7)  # 设置透明度
    plt.show()
