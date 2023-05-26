import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from sklearn import manifold

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def draw(train_result_path, val_result_path, epochs=100):
    # 读取结果数据
    train_result = np.loadtxt(train_result_path, delimiter=',').tolist()
    val_result = np.loadtxt(val_result_path, delimiter=',').tolist()

    # 画train_loss曲线
    plt.figure()
    plt.plot(range(epochs), train_result[0], 'r', linestyle='-')
    plt.title('Training loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Train Loss"])
    plt.show()

    # 画train_accuracy曲线
    plt.figure()
    plt.plot(range(epochs), train_result[1], 'r', linestyle='-')
    plt.plot(range(epochs), train_result[2], 'b', linestyle='dashdot')
    plt.plot(range(epochs), train_result[3], 'g', linestyle='dotted')
    plt.title('Training accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Type", "Location", "Diameter"])
    plt.show()

    # 画val_loss曲线
    plt.figure()
    plt.plot(range(epochs), val_result[0], 'r', linestyle='-')
    plt.title('Validate loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Validation Loss"])
    plt.show()

    # 画val_accuracy曲线
    plt.figure()
    plt.plot(range(epochs), val_result[1], 'r', linestyle='-')
    plt.plot(range(epochs), val_result[2], 'b', linestyle='dashdot')
    plt.plot(range(epochs), val_result[3], 'g', linestyle='dotted')
    plt.title('Validate accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Type", "Location", "Diameter"])
    plt.show()


def confusion_matrix(txt_path, jpg_path):
    # 加载混淆矩阵
    con_matrix = np.loadtxt(txt_path, delimiter=',')
    # 标签
    classes = ['Norm', 'B07', 'B14', 'B21', 'IR07', 'IR14', 'IR21', 'OR07', 'OR14', 'OR21']
    # 标签的个数
    classNamber = 10  # 分类数量
    # 按行进行归一化到(0,1)之间
    con_matrix = torch.tensor(con_matrix)
    con_matrix = F.normalize(con_matrix.float(), p=1, dim=-1)
    con_matrix = con_matrix.numpy()

    plt.imshow(con_matrix, interpolation='nearest', cmap=plt.cm.GnBu)  # 按照像素显示出矩阵,可设置像素背景颜色
    plt.title('LSTM')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-45)
    plt.yticks(tick_marks, classes)

    # ij配对,遍历矩阵迭代器,将矩阵按元素一个个填入格子
    iters = np.reshape([[[i, j] for j in range(classNamber)] for i in range(classNamber)], (con_matrix.size, 2))
    for i, j in iters:
        plt.text(j, i, '{:.3f}'.format(con_matrix[i, j]), fontsize=6, va='center', ha='center')  # 显示对应的数字

    plt.ylabel('Real label')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig(jpg_path)
    plt.show()


def estimate(estimate_path):
    fd_acc = []
    for i in range(5):
        fs = open(estimate_path + '/exp0{}/test_result.txt'.format(i+1))
        data = fs.read().split('\n')
        fd_acc.append(eval(data[0]))
        fs.close()

    fd_var = np.var(fd_acc)
    fd_mean = np.mean(fd_acc)
    result_fd = {'fd均值: ': f'{fd_mean:.2f}',
                 'fd方差: ': f'{fd_var:.2f}'}

    fs = open(estimate_path + '/estimate.txt', 'w')
    fs.write(str(result_fd))
    fs.close()


def tsne(data_tsne, label, class_num):
    # 读取数据
    features = torch.Tensor(data_tsne)
    labels = torch.Tensor(label)

    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_ts = ts.fit_transform(features)  # [num, 2]
    x_min, x_max = x_ts.min(0), x_ts.max(0)
    x_final = (x_ts - x_min) / (x_max - x_min)

    true_labels = labels.reshape((-1, 1))
    s_data = np.hstack((x_final, true_labels))  # 将降维后的特征与相应的标签拼接在一起
    s_data = pd.DataFrame({'x': s_data[:, 0], 'y': s_data[:, 1], 'label': s_data[:, 2]})  # [num, 3]

    plt.figure(figsize=(7, 7))
    # 设置散点形状
    maker = ['.', 'o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']

    # 设置散点颜色
    colors = ['#104680', '#B0A875', '#FC8002', '#369F2D', '#EE4431', '#614099', '#1663A9', '#B9181A', '#B4B4D5',
              '#36ACA2']

    # 图例名称
    label_fd = ['Norm', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']

    # 设置字体格式
    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 32,
             }

    for index in range(class_num):  # 假设总共有class_num个类别
        x = s_data.loc[s_data['label'] == index]['x']
        y = s_data.loc[s_data['label'] == index]['y']
        plt.scatter(x, y, cmap='brg', s=100, marker=maker[0], c=colors[index], edgecolors=colors[index], alpha=1)
        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值

    plt.show()


if __name__ == '__main__':
    group_index = 4
    for i in range(5):
        txt_path = 'result/result_cu_noisy/group{}/exp0{}/confusion_matrix.txt'.format(group_index, i + 1)
        jpg_path = 'result/result_cu_noisy/group{}/exp0{}/confusion_matrix.jpg'.format(group_index, i + 1)
        confusion_matrix(txt_path, jpg_path)

        train_result_path = 'result/result_cu_noisy/group{}/exp0{}/train_result.txt'.format(group_index, i + 1)
        val_result_path = 'result/result_cu_noisy/group{}/exp0{}/val_result.txt'.format(group_index, i + 1)
        #draw(train_result_path, val_result_path, epochs=100)
    estimate_path = 'result/result_cu_noisy/group{}'.format(group_index)
    estimate(estimate_path)

    # tsne
    # group_index = 4
    # fd_tsne = np.loadtxt('result/result_cu_noisy/group{}/exp0{}/fd_tsne.txt'.format(group_index, 1), delimiter=',')
    # labels = np.loadtxt('result/result_cu_noisy/group{}/exp0{}/labels_tsne.txt'.format(group_index, 1), delimiter=',')
    # tsne(fd_tsne, labels, 10)

