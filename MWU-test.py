import seaborn as sns
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

model_seed = 12345
seed = 1
test_size = 0.2
hidden_channels = 15
train_batch_size = 80
test_batch_size = 7
learing_rate = 0.008
epochs = 20
a_lambda = 0.1

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



class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(model_seed)
        self.k = 70
        self.conv1 = GCNConv(90, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels * self.k, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight)
        x_train = x
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)

        # 2. Readout layer
        x = global_sort_pool(x, batch, self.k)  # [batch_size, hidden_channels*self.k]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.sigmoid(x)
        x = torch.squeeze(x)

        return x, x_train

device = 'cpu'

model = GCN(hidden_channels=hidden_channels).to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCELoss()

def dis2(x, label, N1, N2):

    batch_sample = np.vsplit(x.detach().cpu().numpy(), int(x.detach().cpu().numpy().shape[0]) / 90)
    sample = []
    for i in range(len(batch_sample)):
        ls = batch_sample[i]
        ls_t = mat(ls).T
        result = mat(ls) * ls_t  # 90*90
        sample.append(result)
    return sample

def train():
    model.train()
    loss_epoch = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        out, x_train = model(data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device),
                             data.batch.to(device))  # Perform a single forward pass. data.x 4500*15
        N1 = int((1 == data.y).sum())
        N2 = int((0 == data.y).sum())
        sample = dis2(x_train, data.y.to(device), N1, N2)
        loss = criterion(out, data.y.float().to(device))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    print('------------Finished Training------------')

    PATH = './ks_test.pth'
    torch.save(model.state_dict(), PATH)
    return model, PATH, sample


OUT_TOTAL = []
n=0
a=0
for i in range(0, 10):
    test_list = data_list[n*14:(n+1)*14]
    train_list = [val for val in data_list if val not in test_list]
    n=n+1
    train_loader = DataLoader(train_list, batch_size=train_batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=test_batch_size, shuffle=False)
    test_total_acc = []
    model, PATH,sample = train()
    count1 = 0
    count0 = 0
    d1 = []
    d0 = []

    sample_adj = np.array(sample)
    test_label = data_y[a * 14:(a + 1) * 14]
    train_label = concatenate((data_y[0:a * 14],data_y[(a+1) * 14:137]), axis=0)
    a = a + 1
    sample_label = train_label
    a = 0
    for i in range(len(sample_adj)):
        sample_deg = np.diag(np.sum(sample_adj[i], axis=1))
        sample_lap = sample_deg - sample_adj[i]
        sample_eigvalue = np.linalg.eig(sample_lap)[0]
        sam_label=sample_label[a]
        if sam_label == 1:
            count1 += 1
            d1.append(sample_eigvalue)
        else:
            count0 += 1
            d0.append(sample_eigvalue)
        a=a+1
    d0x = []
    d1x = []
    for i in range(90):
        d0x.append([])
        for j in range(count0):
            d0x[i].append(d0[j][i])
    for i in range(90):
        d1x.append([])
        for j in range(count1):
            d1x[i].append(d1[j][i])
print("DTI+fMRI result")

datap=[d0x[1], d1x[1]]

plt.figure(figsize=(15,8))
plt.style.use('default')
sns.kdeplot(datap[0],color='coral',linewidth = 2.5)
sns.kdeplot(datap[1],color='steelblue',linewidth = 1.5)
plt.legend(['Positive','Negative'])
plt.title('Eigenvalue Kernel Density Graph of Samples')
plt.savefig('KS-test.')

plt.figure(figsize=(180,120))
for i in range(90):
    plt.subplot(9,10, i+1)
    p1=sns.kdeplot(d0x[i], color="coral",linewidth = 6.5)
    p1=sns.kdeplot(d1x[i], color="steelblue",linewidth = 5.5)
    plt.legend(['Positive','Negative'])
    plt.rc('legend',fontsize=20)
    plt.title(i+1,y=-0.1,fontsize=60)

plt.savefig('KS-test.',dpi=300)

d0x = np.array(d0x)
d1x = np.array(d1x)

p_value = []
d_value = []
result = []

for i in range(90):
    p_value.append(np.average(stats.mannwhitneyu(d0x[i], d1x[i])[1]))
    d_value.append(np.average(stats.mannwhitneyu(d0x[i], d1x[i])[0]))

for i, v in enumerate(p_value):
    result.append((i, v, d_value[i]))
result.sort(key=lambda x: x[1], reverse=True)
result1 = pd.DataFrame(result, columns=["node", "p-value", "d-value"])

result1.to_csv("KS-test.csv")
