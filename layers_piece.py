import sys
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import tanh
import time
from torch import bfloat16
from utils import *


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class fc_layer(nn.Module):

    def __init__(self, embedding_dim, out_size):
        super(fc_layer, self).__init__()
        self.predictor = nn.Linear(embedding_dim, out_size)

    def forward(self, input_feature):
        pre = self.predictor(input_feature)
        return pre


class FFN_Network(nn.Module):
    def __init__(self, hidden_size, ):
        super(FFN_Network, self).__init__()
        self.layer1 = nn.Linear(input_size=hidden_size, hidden_size=hidden_size * 3)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(input_size=hidden_size * 3, hidden_size=hidden_size)

    def forward(self, feature):
        feature = self.layer1(feature)
        feature = self.activation(feature)
        feature = self.layer2(feature)
        return feature


class simple_LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        super(simple_LSTM, self).__init__()
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1)

    def forward(self, seq):
        output, (hidden, cell) = self.encoder(seq.to(torch.float32))
        return hidden


class Graph_Tensor(nn.Module):
    def __init__(self, num_stock, d_hidden, d_market, d_piece, device, bias=True):
        super(Graph_Tensor, self).__init__()
        self.num_stock = num_stock
        self.d_hidden = d_hidden
        self.d_market = d_market
        self.seq_transformation_markets_1 = nn.Conv1d(d_market, d_hidden, kernel_size=1, stride=1, bias=False)
        self.seq_transformation_markets_2 = nn.Conv1d(d_market, d_hidden, kernel_size=1, stride=1, bias=False)

        self.tensorGraph_relation1 = nn.Parameter(torch.zeros(1, d_piece, d_hidden, d_hidden))
        self.mapping_1 = nn.Parameter(torch.zeros(1, d_piece, d_hidden, d_hidden))
        self.mapping_2 = nn.Parameter(torch.zeros(1, d_piece, d_hidden, d_hidden))
        self.mapping_3 = nn.Parameter(torch.zeros(1, d_piece, d_hidden, d_hidden))

        self.W1 = nn.Parameter(torch.zeros(d_hidden, int(d_hidden / 2)))
        self.W2 = nn.Parameter(torch.zeros(d_hidden, int(d_hidden / 2)))
        self.W3 = nn.Parameter(torch.zeros(d_piece, d_hidden))
        self.b1 = nn.Parameter(torch.zeros(num_stock, d_hidden))

        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, market, adj_num, cluster_info):
        day_num, num_stocks = 1, market.size()[0]

        news_transformed = market.reshape(-1, self.d_market)  # 合并天数和股票数的维度
        news_transformed = torch.transpose(news_transformed, 0, 1).unsqueeze(0)  # -->dim(1,d_market,天数*股票数)
        news_transformed = self.seq_transformation_markets_1(news_transformed)  # 线性变换 -->dim(1,d_hidden,天数*股票数)
        news_transformed = news_transformed.squeeze().transpose(0, 1)  # -->dim(天数*股票数,d_hidden)
        news_transformed = news_transformed.reshape(num_stocks, self.d_hidden)  # -->dim(股票数,d_hidden)

        market_transformed = market.reshape(-1, self.d_market)
        market_transformed = torch.transpose(market_transformed, 0, 1).unsqueeze(0)
        market_transformed = self.seq_transformation_markets_2(market_transformed)
        market_transformed = market_transformed.squeeze().transpose(0, 1)
        market_transformed = market_transformed.reshape(num_stocks, self.d_hidden)

        x_news_tensor = news_transformed.unsqueeze(1).unsqueeze(1)  # -->dim(股票数,1,d_hidden)
        x_market_tensor = market_transformed.unsqueeze(0).expand(num_stocks, num_stocks,
                                                                 self.d_hidden)  # -->dim(股票数,d_hidden,1)


        output = torch.zeros(len(adj_num.keys()), num_stocks, self.d_hidden, device=market.device)

        for num in range(0, len(adj_num.keys())):
            if (num == 0):
                temp_tensor = x_news_tensor.matmul(
                    self.tensorGraph_relation1.mul(F.relu(self.mapping_1))).squeeze()  # -->dim(股票数,d_hidden,d_hidden)
            elif (num == 1):
                temp_tensor = x_news_tensor.matmul(
                    self.tensorGraph_relation1.mul(F.relu(self.mapping_2))).squeeze()  # -->dim(股票数,d_hidden,d_hidden)
            elif (num == 2):
                temp_tensor = x_news_tensor.matmul(
                    self.tensorGraph_relation1.mul(F.relu(self.mapping_3))).squeeze()  # -->dim(股票
            temp_tensor = temp_tensor.permute(0, 2, 1)
            adj = adj_num[str(num + 1)]
            x_market_tensor = x_market_tensor.mul(adj.unsqueeze(-1))
            final_tensor = x_market_tensor.matmul(temp_tensor).sum(1).squeeze()
            final_tensor = final_tensor.matmul(self.W3)
            final_linear_1 = torch.mm(news_transformed, self.W1)
            final_linear_2 = torch.mm(market_transformed, self.W2)
            final_linear_2 = final_linear_2.expand(num_stocks, num_stocks, int(self.d_hidden / 2)).mul(
                adj.unsqueeze(-1)).sum(1).squeeze()
            final_linear = torch.cat((final_linear_1, final_linear_2), axis=-1)
            output[num] = torch.relu(final_tensor + final_linear + self.b1)

        return output


class Graph_Attention(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, device, num_stock, head_size, concat=True,
                 residual=False):
        super(Graph_Attention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.residual = residual
        self.head_size = head_size
        self.num_stock = num_stock
        # W参数矩阵
        #         self.createVar1 = locals()
        self.seq_transformation_s = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)

        self.query1 = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        self.b_query1 = nn.Parameter(torch.zeros(num_stock, out_features))
        self.key1 = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        self.b_key1 = nn.Parameter(torch.zeros(num_stock, out_features))
        self.value1 = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        self.b_value1 = nn.Parameter(torch.zeros(num_stock, out_features))

        self.f_1 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.f_2 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.scale = torch.sqrt(torch.FloatTensor([self.out_features])).to(device)

        self.ffn = nn.Linear(out_features, out_features)

        if self.residual:
            self.proj_residual = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)

        self.coef_revise = False
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def get_relation(self, stock_feature, adj, cluster_info, i, num_stock):

        seq_r = torch.transpose(stock_feature, 0, 1).unsqueeze(0)

        query_r = self.query1(seq_r)
        query_r = torch.transpose(query_r, 2, 1)
        query_r = query_r + self.b_query1
        key_r = self.key1(seq_r)
        key_r = torch.transpose(key_r, 2, 1)
        key_r = key_r + self.b_key1

        weight_matrix = torch.zeros(num_stock, num_stock, device=stock_feature.device, dtype=torch.float16)
        f_1 = self.f_1(torch.transpose(query_r, 1, 2))
        f_2 = self.f_2(torch.transpose(key_r, 1, 2))
        weight_matrix += (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)

        weight_matrix = weight_matrix.mul(adj)
        weight_matrix = weight_matrix / self.scale
        return weight_matrix

    def forward(self, input_s, input_r, adj, cluster_info, num_stock):  # !!!!!!!!!!!!!!!input_s, input_r,
        ret = torch.zeros(len(adj.keys()), num_stock, self.out_features, device=input_s.device)
        for i in range(0, len(adj.keys())):
            # unmasked attention
            seq_r = torch.transpose(input_s[i], 0, 1).unsqueeze(0)

            value_r = self.value1(seq_r)
            value_r = torch.transpose(value_r, 2, 1).squeeze(0)
            value_r = value_r + self.b_value1
            # TODO: 对角元素归零
            temp_feature = torch.zeros(self.num_stock, self.out_features, device=input_s.device)
            # for num in range(self.head_size):
            adj_ = adj[str(i + 1)] + torch.eye(adj[str(i + 1)].shape[0], device=input_s.device)
            adj_ = torch.where(adj_ > 1, adj_ - 1, adj_)
            coefs_eye = self.get_relation(input_s[i], adj_, cluster_info, i, num_stock)
            # temp_adj = adj[str(i + 1)] + torch.eye(adj[str(i + 1)].shape[0], device=input_s.device)
            temp_index = torch.nonzero(adj_ == 0, as_tuple=True)
            coefs_eye[temp_index[0], temp_index[1]] = -11111
            coefs_eye = F.softmax(coefs_eye, dim=-1)  # ！！！！！
            coefs_eye = coefs_eye.to(torch.float16)
            value_r = value_r.to(torch.float16)
            temp_ = torch.mm(coefs_eye,
                             value_r)
            temp_feature[:] = temp_
                # ret[i] += torch.mm(coefs_eye, value_r).squeeze()
            ret[i] = self.ffn(temp_feature).squeeze()
        final = ret.sum(0).squeeze()
        return final


class GraphConvolution(nn.Module):  ##图卷积层
    def __init__(self, d_markets, rnn_hidden_size, GNN_hidden_size, GNN_output_size, d_piece, num_stock, num_layers,
                 device, head_size,
                 use_bias=True, gau_flag=False):
        super(GraphConvolution, self).__init__()  # 调用父类的构造方法
        self.GNN_hidden_size = GNN_hidden_size
        self.GNN_output_size = GNN_output_size
        self.use_bias = use_bias
        self.num_stock = num_stock
        self.num_layers = num_layers
        self.head_size = head_size
        # self.Graph_Attention = Graph_Attention(GNN_hidden_size, GNN_hidden_size, 0.5, 0.5, device, num_stock, head_size)
        self.Graph_Attention = [
            Graph_Attention(GNN_hidden_size, GNN_hidden_size, 0.5, 0.5, device, num_stock, head_size) for _
            in range(head_size)]
        for i, attention in enumerate(self.Graph_Attention):
            self.add_module('attention_{}'.format(i), attention)
        self.Graph_Tensor = Graph_Tensor(self.num_stock, GNN_hidden_size, GNN_hidden_size, d_piece, device)
        self.gau = GAU(
            dim=GNN_hidden_size,
            query_key_dim=128,  # query / key dimension
            causal=False,  # autoregressive or not
            expansion_factor=2,  # hidden dimension = dim * expansion_factor
            laplace_attn_fn=False  # new Mega paper claims this is more stable than relu squared as attention function
        )
        self.wMatrix_sl = nn.Linear(GNN_hidden_size * head_size, GNN_hidden_size)
        # self.wMatrix_sl = nn.Parameter(torch.Tensor(rnn_hidden_size, GNN_hidden_size).requires_grad_(True))  # 8为股票节点特

        self.coef_revise = False
        self.gau_flag = gau_flag

        if self.use_bias:
            self.bias1 = nn.Parameter(torch.Tensor(GNN_hidden_size))
            self.bias2 = nn.Parameter(torch.Tensor(GNN_output_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_bias:
            init.zeros_(self.bias1)
            init.zeros_(self.bias2)

    def next_layer(self, stock_feature, Adj_dict, cluster_info):
        weight_adjacency = Adj_dict  # 未归一化之前的传播权重
        temp_feature = self.Graph_Tensor(stock_feature.squeeze(), weight_adjacency, cluster_info)
        if not self.gau_flag:
            ret = torch.cat([att(temp_feature, temp_feature, weight_adjacency, cluster_info,
                                                self.num_stock, ) for att in self.Graph_Attention], dim=1)
            ret = self.wMatrix_sl(ret)
        else:
            # 对角线归零
            for i in range(0, len(weight_adjacency.keys())):
                ret = torch.zeros(len(weight_adjacency.keys()), self.num_stock, self.GNN_hidden_size, device=temp_feature.device)
                adj_ = weight_adjacency[str(i+1)] + torch.eye(weight_adjacency[str(i+1)].shape[0], device=temp_feature.device)
                adj_ = torch.where(adj_ > 1, adj_ - 1, adj_)
                adj_ = adj_ - torch.eye(weight_adjacency[str(i+1)].shape[0], device=temp_feature.device)
                ret[i] = self.gau(temp_feature[[i]], mask=adj_)
            ret = ret.sum(0).squeeze()
        final_stock_feature = torch.relu(ret)
        return final_stock_feature

    def forward(self, adj, feats, cluster_info,):
        h = feats
        for layer in range(2):
            h = self.next_layer(h, Adj_dict=adj, cluster_info=cluster_info).squeeze()
            h = h + feats.squeeze()

        return h
