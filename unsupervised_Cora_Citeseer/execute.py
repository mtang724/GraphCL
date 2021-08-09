import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random
from models import DGI, LogReg
from utils import process
import pdb
import aug
import os
import argparse
import statistics

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# MLP with linear outputs (without softmax)
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)



class NodeClassificationDataset(Dataset):
    def __init__(self, node_embeddings, labels):
        self.len = node_embeddings.shape[0]
        self.x_data = node_embeddings
        self.y_data = labels

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def train_real_datasets(emb, node_labels, nb_classes):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(emb.shape)
    class_number = nb_classes
    input_dims = emb.shape
    FNN = MLP(num_layers=4, input_dim=input_dims[1], hidden_dim=input_dims[1] // 2, output_dim=class_number).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(FNN.parameters())
    dataset = NodeClassificationDataset(emb, node_labels)
    split = process.DataSplit(dataset, shuffle=True)
    train_loader, val_loader, test_loader = split.get_split(batch_size=64, num_workers=0)
    print(len(train_loader.dataset))
    print(len(node_labels), emb.shape)
    # train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    best = float('inf')
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            # data = data.to(device)
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            y_pred = FNN(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                correct = 0
                total = 0
                for data in val_loader:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = FNN(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    total += labels.size(0)
                    correct += torch.sum(predicted == labels)
            if loss < best:
                best = loss
                torch.save(FNN.state_dict(), 'best_mlp.pkl')
            # print(str(epoch), correct / total)

    with torch.no_grad():
        FNN.load_state_dict(torch.load('best_mlp.pkl'))
        correct = 0
        total = 0
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = FNN(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels)
    # print((correct / total).item())
    return torch.true_divide(correct, total).item()

parser = argparse.ArgumentParser("My DGI")

parser.add_argument('--dataset',          type=str,           default="texas",                help='data')
parser.add_argument('--aug_type',         type=str,           default="subgraph",                help='aug type: mask or edge')
parser.add_argument('--drop_percent',     type=float,         default=0.1,               help='drop percent')
parser.add_argument('--seed',             type=int,           default=39,                help='seed')
parser.add_argument('--gpu',              type=int,           default=0,                 help='gpu')
parser.add_argument('--save_name',        type=str,           default='try.pkl',                help='save ckpt name')

args = parser.parse_args()

print('-' * 100)
print(args)
print('-' * 100)

dataset = args.dataset
aug_type = args.aug_type
drop_percent = args.drop_percent
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# training params


batch_size = 1
nb_epochs = 10000
patience = 20
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 512
sparse = True


nonlinearity = 'prelu' # special name to separate parameters
# adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
adj, features, labels = process.read_real_datasets(dataset)
print(labels)
features, _ = process.preprocess_features(features)

nb_nodes = features.shape[0]  # node number
ft_size = features.shape[1]   # node features dim
nb_classes = int(max(labels)) + 1  # classes = 6
print(nb_classes)

features = torch.FloatTensor(features[np.newaxis])


'''
------------------------------------------------------------
edge node mask subgraph
------------------------------------------------------------
'''
print("Begin Aug:[{}]".format(args.aug_type))
if args.aug_type == 'edge':

    aug_features1 = features
    aug_features2 = features

    aug_adj1 = aug.aug_random_edge(adj, drop_percent=drop_percent) # random drop edges
    aug_adj2 = aug.aug_random_edge(adj, drop_percent=drop_percent) # random drop edges
    
elif args.aug_type == 'node':
    
    aug_features1, aug_adj1 = aug.aug_drop_node(features, adj, drop_percent=drop_percent)
    aug_features2, aug_adj2 = aug.aug_drop_node(features, adj, drop_percent=drop_percent)
    
elif args.aug_type == 'subgraph':
    
    aug_features1, aug_adj1 = aug.aug_subgraph(features, adj, drop_percent=drop_percent)
    aug_features2, aug_adj2 = aug.aug_subgraph(features, adj, drop_percent=drop_percent)

elif args.aug_type == 'mask':

    aug_features1 = aug.aug_random_mask(features,  drop_percent=drop_percent)
    aug_features2 = aug.aug_random_mask(features,  drop_percent=drop_percent)
    
    aug_adj1 = adj
    aug_adj2 = adj

else:
    assert False



'''
------------------------------------------------------------
'''

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    sp_aug_adj1 = process.sparse_mx_to_torch_sparse_tensor(aug_adj1)
    sp_aug_adj2 = process.sparse_mx_to_torch_sparse_tensor(aug_adj2)

else:
    adj = (adj + sp.eye(adj.shape[0])).todense()
    aug_adj1 = (aug_adj1 + sp.eye(aug_adj1.shape[0])).todense()
    aug_adj2 = (aug_adj2 + sp.eye(aug_adj2.shape[0])).todense()


'''
------------------------------------------------------------
mask
------------------------------------------------------------
'''

'''
------------------------------------------------------------
'''
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
    aug_adj1 = torch.FloatTensor(aug_adj1[np.newaxis])
    aug_adj2 = torch.FloatTensor(aug_adj2[np.newaxis])


# labels = torch.FloatTensor(labels[np.newaxis])
# idx_train = torch.LongTensor(idx_train)
# idx_val = torch.LongTensor(idx_val)
# idx_test = torch.LongTensor(idx_test)

model = DGI(ft_size, hid_units, nonlinearity)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    aug_features1 = aug_features1.cuda()
    aug_features2 = aug_features2.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
        sp_aug_adj1 = sp_aug_adj1.cuda()
        sp_aug_adj2 = sp_aug_adj2.cuda()
    else:
        adj = adj.cuda()
        aug_adj1 = aug_adj1.cuda()
        aug_adj2 = aug_adj2.cuda()

    labels = labels.cuda()
    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

for epoch in range(nb_epochs):

    model.train()
    optimiser.zero_grad()

    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]

    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()
    
    logits = model(features, shuf_fts, aug_features1, aug_features2,
                   sp_adj if sparse else adj, 
                   sp_aug_adj1 if sparse else aug_adj1,
                   sp_aug_adj2 if sparse else aug_adj2,  
                   sparse, None, None, None, aug_type=aug_type) 

    loss = b_xent(logits, lbl)
    print('Loss:[{:.4f}]'.format(loss.item()))

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), args.save_name)
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()

print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load(args.save_name))

embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
embeddings = torch.squeeze(embeds.cpu().detach())
acc = []
for i in range(5):
    result = train_real_datasets(embeddings, labels, nb_classes)
    acc.append(result)
print("mean:")
print(statistics.mean(acc))
print("std:")
print(statistics.stdev(acc))
# train_embs = embeds[0, idx_train]
# val_embs = embeds[0, idx_val]
# test_embs = embeds[0, idx_test]
#
# train_lbls = torch.argmax(labels[0, idx_train], dim=1)
# val_lbls = torch.argmax(labels[0, idx_val], dim=1)
# test_lbls = torch.argmax(labels[0, idx_test], dim=1)
#
# tot = torch.zeros(1)
# tot = tot.cuda()
#
# accs = []
#
# for _ in range(50):
#     log = LogReg(hid_units, nb_classes)
#     opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
#     log.cuda()
#
#     pat_steps = 0
#     best_acc = torch.zeros(1)
#     best_acc = best_acc.cuda()
#     for _ in range(100):
#         log.train()
#         opt.zero_grad()
#
#         logits = log(train_embs)
#         loss = xent(logits, train_lbls)
#
#         loss.backward()
#         opt.step()
#
#     logits = log(test_embs)
#     preds = torch.argmax(logits, dim=1)
#     acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
#     accs.append(acc * 100)
#     print('acc:[{:.4f}]'.format(acc))
#     tot += acc
#
# print('-' * 100)
# print('Average accuracy:[{:.4f}]'.format(tot.item() / 50))
# accs = torch.stack(accs)
# print('Mean:[{:.4f}]'.format(accs.mean().item()))
# print('Std :[{:.4f}]'.format(accs.std().item()))
# print('-' * 100)


