import torch
import os
import numpy as np
from model import RegLSTM
from tqdm import tqdm
from data_set import MyDataset
from torch.utils.data import DataLoader
from einops import rearrange

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def tsne_data(tsne_list, labels, save_path):
    fd_tsne = tsne_list.cpu().numpy()

    np.savetxt(save_path + 'fd_tsne.txt', fd_tsne, fmt='%.5f', delimiter=',')
    np.savetxt(save_path + 'labels_tsne.txt', labels.cpu().numpy(), fmt='%.5f', delimiter=',')

def prediction(weights_path, con_matrix_path, acc_path, tsne_save_path, sne=False):
    #定义参数
    N = 4 #编码器个数
    input_dim = 1024
    seq_len = 16 #句子长度
    hidden_size = 128
    #batch_size = 64
    batch_size = 814 #tsne

    test_path = r'F:\PyCharmWorkSpace\MultiFD\data\cu_data\test\test.csv'
    test_dataset = MyDataset(test_path, 'fd')
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    #定义模型
    model = RegLSTM(input_dim=input_dim, input_size=seq_len, hidden_size=hidden_size, hidden_num_layers=N, sne=sne)
    model.to(device)

    # load model weights
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    acc_fd = []
    con_matrix = torch.zeros(10, 10).type(torch.LongTensor).to(device) # 自己的数据是7分类
    test_bar = tqdm(test_loader)
    for datas, labels in test_bar:
        datas, labels = datas.to(device), labels.to(device)
        with torch.no_grad():
            #outputs = model(datas.float().to(device))

            outputs, tsne_list = model(datas.float().to(device))
            tsne_data(tsne_list, labels, tsne_save_path)

        acc_fd.append((outputs.argmax(dim=-1) == labels).float().mean())
        #生成混淆矩阵
        for i in range(labels.size(0)):
            vect = torch.zeros(1, 10).type(torch.LongTensor).to(device) # 自己的数据是7分类
            temp = torch.max(outputs, dim=1)[1]
            vect[0, temp[i]] = 1
            index = labels[i].item()
            con_matrix[int(index)] = con_matrix[int(index)] + vect

    print(f'Test acc_fd = {(sum(acc_fd) / len(acc_fd)).item():.5f}')
    np.savetxt(con_matrix_path, con_matrix.cpu(), fmt='%d', delimiter=',')
    fs = open(acc_path, 'w')
    fs.write(f'{(sum(acc_fd) / len(acc_fd) * 100.0).item():.5f}')
    fs.close()


if __name__ == '__main__':
    # group_index = 4
    # for i in range(5):
    #     weights_path = "result/result_cu_noisy/group{}/exp0{}/model.pth".format(group_index, i + 1)
    #     con_matrix_path = "result/result_cu_noisy/group{}/exp0{}/confusion_matrix.txt".format(group_index, i + 1)
    #     acc_path = "result/result_cu_noisy/group{}/exp0{}/test_result.txt".format(group_index, i + 1)
    #     tsne_save_path = "result/result_cu_noisy/group{}/exp0{}/".format(group_index, i + 1)
    #     prediction(weights_path, con_matrix_path, acc_path, tsne_save_path)

    # weights_path = "result/result_own/group{}/exp0{}/model.pth".format(26, 1)
    # con_matrix_path = "result/result_own/group{}/exp0{}/confusion_matrix.txt".format(26, 1)
    # acc_path = "result/result_own/group{}/exp0{}/test_result.txt".format(26, 1)
    # tsne_save_path = "result/result_own/group{}/exp0{}/".format(26, 1)
    # prediction(weights_path, con_matrix_path, acc_path, tsne_save_path)

    # t-sne
    group_index = 1
    weights_path = "result/result_cu/group{}/exp0{}/model.pth".format(group_index, 1)
    con_matrix_path = "result/result_cu/group{}/exp0{}/confusion_matrix.txt".format(group_index, 1)
    acc_path = "result/result_cu/group{}/exp0{}/test_result.txt".format(group_index, 1)
    tsne_save_path = "result/result_cu/group{}/exp0{}/".format(group_index, 1)
    prediction(weights_path, con_matrix_path, acc_path, tsne_save_path, sne=True)