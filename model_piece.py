#!/usr/bin/env python
# coding: utf-8

# In[9]:

import time
from layers_piece import *
import random

class GcnNet(nn.Module):
    def __init__(self, d_markets, rnn_hidden_size, GNN_hidden_size, d_piece, num_stock, num_layers, device,
                 dropout=0.2):
        super(GcnNet, self).__init__()
        self.dropout = dropout
        self.lstm = simple_LSTM(d_markets,rnn_hidden_size)
        self.gcn1 = GraphConvolution(d_markets=d_markets,
                                     rnn_hidden_size=rnn_hidden_size,
                                     GNN_hidden_size=GNN_hidden_size,
                                     GNN_output_size=GNN_hidden_size,
                                     d_piece=d_piece,
                                     num_stock=num_stock,
                                     device=device,
                                     num_layers=num_layers,
                                     head_size=2,
                                     )

        self.wMatrix_pred = nn.Parameter(torch.Tensor(GNN_hidden_size,
                                                      GNN_hidden_size).requires_grad_(True))
        self.bias = nn.Parameter(torch.Tensor(GNN_hidden_size).requires_grad_(True))
        self.fc1 = fc_layer(GNN_hidden_size, 2)
        self.reset_parameters()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(GNN_hidden_size)
        self.read = AvgReadout()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, tensor_adjacency, x_train, cluster_info, cluster_num):
        stock_feature = self.lstm(x_train)

        # 分产业链聚合。
        stock_feature = F.dropout(stock_feature, self.dropout, training=self.training)
        h = self.gcn1(tensor_adjacency, stock_feature, cluster_info)
        loss_global = 0
        criterion = nn.BCEWithLogitsLoss()

        for i in range(cluster_num):
            node_idx = cluster_info[i]
            h_1_block = torch.unsqueeze(h[node_idx], 0)
            c_block = self.read(h_1_block, msk=None)
            c_block = self.sigm(c_block)
            random_index_list = []
            for _ in range(len(node_idx)):
                random_supply_index = random.randint(0, cluster_num - 1)
                while (random_supply_index == i):
                    random_supply_index = random.randint(0, cluster_num - 1)
                random_node_index = random.choice(cluster_info[random_supply_index])
                random_index_list.append(random_node_index)
            h_2_block = torch.unsqueeze(h[random_index_list], 0)

            lbl_1 = torch.ones(1, len(node_idx), device=stock_feature.device)
            lbl_2 = torch.zeros(1, len(node_idx), device=stock_feature.device)
            lbl = torch.cat((lbl_1, lbl_2), 1)

            # loss of global information
            ret = self.disc(c_block, h_1_block, h_2_block, s_bias1=None, s_bias2=None)
            loss_tmp = criterion(ret, lbl)
            loss_global += loss_tmp
        loss_global = loss_global / cluster_num
        h = F.dropout(h, self.dropout, training=self.training)

        h = F.tanh(self.fc1(h))
        outcome = F.log_softmax(h, dim=1)
        return outcome, loss_global

# In[ ]:
