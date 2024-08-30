import numpy as np
import torch
import os
import scipy
import pandas as pd
from scipy import stats,linalg
from numpy import linalg as la
import torch.nn.functional as F
from torch.nn import Linear,Conv2d, MaxPool2d,ReLU
from torch_geometric.transforms import OneHotDegree as ohd
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_sort_pool, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from numpy import random,mat
import networkx as nx
import numpy.linalg as lg
import copy
import scipy.linalg as slg
from ipywidgets import interact
from stochastic import *
from sklearn.metrics import roc_curve,auc,classification_report
from numpy import *


import argparse
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--ts', type=float, default=0.3)
parser.add_argument('--hc', type=int, default=15)
parser.add_argument('--trbs', type=int, default=69)
parser.add_argument('--tebs', type=int, default=9)
parser.add_argument('--lr', type=float, default=8e-4)
args = parser.parse_args()

model_seed = 12345
seed = args.seed
test_size = args.ts
hidden_channels = args.hc
train_batch_size = args.trbs
test_batch_size = args.tebs
learing_rate = args.lr
epochs = 20

np.random.seed(seed)
torch.random.manual_seed(seed)
random.seed(seed)

dti ='./huaxi/graph/data_hx/DTI'
fmri = './huaxi/graph/data_hx/fMRI'

def get_max_degree(filepath):
    degree_graph = []
    for filename in os.listdir(filepath):
        sample_path = filepath + '/' + filename  # ./data/DTI//N10001.txt
        sample_adj = np.loadtxt(sample_path)  
        torch_adj = torch.from_numpy(sample_adj)
        edge_index = (torch_adj > 0).nonzero().t()
        current_max_degree = int(degree(edge_index[0]).max())
        degree_graph.append(current_max_degree)
    return max(degree_graph)

def get_dataset_list(graph_path, fea_path, max_degree):
    data_list = []
    for filename in os.listdir(graph_path):
        digits = ''.join([x for x in filename if x.isdigit()])
        sample_label = int(digits[0])
        sample_label = abs(sample_label - 2)
        sample_id = digits[1:]
        sample_path = graph_path + '/' + filename
        sample_adj = np.loadtxt(sample_path)
        torch_adj = torch.from_numpy(sample_adj)

        for file in os.listdir(fea_path):
            digits_ = ''.join([x for x in file if x.isdigit()])
            sample_id_ = digits_[1:]
            if sample_id == sample_id_:
                sample_fea_path = fea_path + '/' + file
                sample_fea = np.loadtxt(sample_fea_path)
                sample_fea = torch.from_numpy(sample_fea)
                sample_fea = sample_fea.float()
            else:
                continue

        edge_index = (torch_adj > 0).nonzero().t()
        edge_weight = torch_adj.view(torch.numel(torch_adj))
        edge_weight = edge_weight[edge_weight.nonzero()]
        edge_weight = edge_weight.squeeze(1)  # torch.Size([788, 1]) — torch.Size([788])
        edge_weight = edge_weight.float()
        graph = Data(x=sample_fea, edge_index=edge_index, y=torch.tensor(sample_label),
                     edge_weight=edge_weight)

        data_list.append(graph)
    return data_list

max_degree = get_max_degree(dti)
data_list = get_dataset_list(dti, fmri, max_degree)

#resuffle the list
data_list1,data_list2 = train_test_split(data_list, test_size=test_size, random_state=seed)
data_list=data_list1+data_list2

data_y=[]
for i in range(len(data_list)):
    data_y.append(float(data_list[i].y.detach().numpy()))
print("data_y:",data_y)

from torch_geometric.nn import GATConv, ChebConv, GCNConv, SAGEConv
class GraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GraphSAGE, self).__init__()
        self.k = 70
        torch.manual_seed(seed)
        self.conv1 = SAGEConv(in_channels=90,out_channels=hidden_channels)
        self.conv2 = SAGEConv(in_channels=hidden_channels,
                              out_channels=hidden_channels)
        self.lin1 = Linear(hidden_channels * self.k, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data, batch):
        x, edge_index,edge_weight = data.x, data.edge_index,data.edge_weight
        x = self.conv1(x, edge_index)
        x_train=x
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_sort_pool(x, batch, self.k)  # [batch_size, hidden_channels*self.k]

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.sigmoid(x)
        x = torch.squeeze(x)

        return x, x_train


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
lr = 0.0003
num_node_features = 90
num_classes = 2
data = data_list[0].to(device)

model = GraphSAGE(num_node_features, num_classes).to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#loss_function = nn.NLLLoss()
criterion = torch.nn.BCELoss()

def train():
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            out, x_train= model(data,data.batch.to(device))  # Perform a single forward pass. data.x 4500*15
            loss = criterion(out, data.y.float())  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            print(f'Epoch: {epoch:03d}, train_loss: {loss:.4f}')
    print('------------Finished Training------------')

    PATH = './GraphSAGE.pth'
    torch.save(model.state_dict(), PATH)
    return model, PATH

def f (precited,expected):
    res = precited ^ expected
    r = np.bincount(res)
    tp_list = ((precited)&(expected))
    fp_list = (precited&(~expected))
    tp_list=tp_list.tolist()
    fp_list=fp_list.tolist()
    tp=tp_list.count(1)
    fp=fp_list.count(1)
    tn = r[0]-tp
    fn = r[1]-fp
    p=tp/(tp+fp)
    recall = tp/(tp+fn)
    F1=(2*tp)/(2*tp+fn+fp)
    acc=(tp+tn)/(tp+tn+fp+fn)
    return tn,fp,recall

OUT_TOTAL = []
n=0
for i in range(0, 10):
    test_list = data_list[n*14:(n+1)*14]
    train_list = [val for val in data_list if val not in test_list]
    n=n+1
    train_loader = DataLoader(train_list, batch_size=train_batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=test_batch_size, shuffle=False)
    test_total_acc = []
    model, PATH = train()

    model.load_state_dict(torch.load(PATH))
    from sklearn.metrics import roc_curve, auc, classification_report,roc_auc_score,recall_score
    from numpy import *

    with torch.no_grad():
        model.eval()
        correct = 0
        for data in test_loader:  # Iterate in batches over the training/test dataset.
            out, _ = model(data,data.batch.to(device))
            OUT_TOTAL = concatenate((OUT_TOTAL, out.numpy()), axis=0)
            print("-------------", "第",len(OUT_TOTAL),"个样本训练完毕", "---------------")



y_true=data_y
y_score=OUT_TOTAL

AUC=roc_auc_score(y_true, y_score)

fpr, tpr, thresh_hold = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

roc_auc = auc(fpr, tpr)
output = open('./graphsage.pkl', 'wb')
pickle.dump([fpr, tpr, roc_auc], output, -1)
output.close()

for i in range(len(y_score)):
    if y_score[i] < 0.5:
        y_score[i] = 0
    else:
        y_score[i] = 1
y_pred = y_score

int_y_true = map(int, y_true)
int_y_pred = map(int, y_pred)

tn, fp, TPR = f(np.array(list(int_y_pred)), np.array(list(int_y_true)))
TNR=tn/(fp+tn)


print("groudtruth:", y_true)
print("predicted labels:", y_score)
print("TNR = %2.10f" % TNR)
print("TPR = %2.10f" % TPR)
print("AUC = %2.10f" % AUC)

with open("GraphSAGE.txt","a") as file:
    file.write("seed = %2.0f" % seed + '\n')
    file.write("model_seed = %2.0f" % model_seed + '\n')
    file.write("test_size = %2.1f" % test_size + '\n')
    file.write("hidden_channels = %2.0f" % hidden_channels + '\n')
    file.write("train_batch_size = %2.0f" % train_batch_size + '\n')
    file.write("test_batch_size = %2.0f" % test_batch_size + '\n')
    file.write("learing_rate = %2.6f" % learing_rate + '\n')
    file.write("epochs = %2.0f" % epochs + '\n')
    file.write('\n')

    file.write("TNR = %2.10f" % TNR + '\n')
    file.write("TPR = %2.10f" % TPR + '\n')
    file.write("AUC = %2.10f" % AUC + '\n')
    file.write("----------------------------------------------------" + '\n')
    file.flush()
