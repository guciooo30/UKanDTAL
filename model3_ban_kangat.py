import dgl
import torch
import numpy as np
import torch.nn as nn
import dgl.function as fn
from BanLayer import BANLayer
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from dgl.nn.functional import edge_softmax
from torch.nn import Sequential, Linear, ReLU
from torch.nn.utils.weight_norm import weight_norm
RDLogger.DisableLog('rdApp.*')

def message_func(edges):
    return {'feat': edges.data['feat']}

def reduce_func(nodes):
    num_edges = nodes.mailbox['feat'].size(1)
    agg_feats = torch.sum(nodes.mailbox['feat'], dim=1) / num_edges
    return {'agg_feats': agg_feats}


def update_node_features(g):
    g.send_and_recv(g.edges(), message_func, reduce_func)
    g.ndata['feat'] = torch.cat((g.ndata['feat'], g.ndata['agg_feats']), dim=1)
    return g

# KA-GAT
class KAN_linear(nn.Module):
    def __init__(self, inputdim, outdim, gridsize, addbias=False):
        super(KAN_linear, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
                                          (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):

        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        # Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        # This should be fused to avoid materializing memory
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        if self.addbias:
            y += self.bias

        y = y.view(outshape)
        return y


class Gat_Kan_layer(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_node_feats, out_edge_feats, num_heads, grid_size, bias=True):
        super(Gat_Kan_layer, self).__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_node = nn.Linear(in_node_feats + in_edge_feats, out_node_feats * num_heads, bias=True)
        self.fc_ni = nn.Linear(in_node_feats, out_edge_feats * num_heads, bias=False)
        self.fc_fij = nn.Linear(in_edge_feats, out_edge_feats * num_heads, bias=False)
        self.fc_nj = nn.Linear(in_node_feats, out_edge_feats * num_heads, bias=False)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_edge_feats)))
        self.output_node = KAN_linear(out_node_feats, out_node_feats, grid_size, addbias=True)
        self.output_edge = KAN_linear(out_edge_feats, out_edge_feats, grid_size, addbias=True)
        self.edge_kan = KAN_linear(out_edge_feats * num_heads, out_edge_feats * num_heads, gridsize=1, addbias=True)
        self.node_kan = KAN_linear(in_node_feats + in_edge_feats, in_node_feats + in_edge_feats, gridsize=1,
                                   addbias=True)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_edge_feats,)))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc_node.weight)
        nn.init.xavier_normal_(self.fc_ni.weight)
        nn.init.xavier_normal_(self.fc_fij.weight)
        nn.init.xavier_normal_(self.fc_nj.weight)
        nn.init.xavier_normal_(self.attn)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def message_func(self, edges):
        return {'feat': edges.data['feat']}

    def reduce_func(self, nodes):
        # 归一化或平均
        num_edges = nodes.mailbox['feat'].size(1)  # 计算接收到的消息的数量
        agg_feats = torch.sum(nodes.mailbox['feat'], dim=1) / num_edges  # 求平均
        return {'agg_feats': agg_feats}

    def forward(self, graph, nfeats, efeats, get_attention=False):
        with graph.local_scope():
            graph.ndata['feat'] = nfeats
            graph.edata['feat'] = efeats
            in_degrees = graph.in_degrees().float().unsqueeze(-1)
            in_degrees[in_degrees == 0] = 1  # 将入度为0的节点设置为1，以避免除零错误
            f_ni = self.fc_ni(nfeats)  # in_node_feats --> out_edge_feats
            f_nj = self.fc_nj(nfeats)  # in_node_feats --> out_edge_feats
            f_fij = self.fc_fij(efeats)  # in_edge_feats --> out_edge_feats

            graph.srcdata.update({'f_ni': f_ni})
            graph.dstdata.update({'f_nj': f_nj})
            graph.apply_edges(fn.u_add_v('f_ni', 'f_nj', 'f_tmp'))

            f_out = graph.edata.pop('f_tmp') + f_fij
            f_out = self.edge_kan(f_out)  # new edge embedding

            if self.bias is not None:
                f_out = f_out + self.bias
            f_out = nn.functional.leaky_relu(f_out)
            f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)

            e = (f_out * self.attn).sum(dim=-1).unsqueeze(-1)

            graph.send_and_recv(graph.edges(), self.message_func, reduce_func=self.reduce_func)
            m_feats = torch.cat((graph.ndata['feat'], graph.ndata['agg_feats']), dim=1)

            m_feats = self.node_kan(m_feats)  # # new node embedding

            graph.edata['a'] = edge_softmax(graph, e)

            graph.ndata['h_out'] = self.fc_node(m_feats).view(-1, self._num_heads, self._out_node_feats)

            graph.update_all(fn.u_mul_e('h_out', 'a', 'm'),
                             fn.sum('m', 'h_out'))

            h_out = nn.functional.leaky_relu(graph.ndata['h_out'])
            h_out = h_out.view(-1, self._num_heads, self._out_node_feats)

            h_out = torch.sum(h_out, dim=1)
            f_out = torch.sum(f_out, dim=1)

            out_n = self.output_node(h_out)
            out_e = self.output_edge(f_out)
            if get_attention:
                return out_n, out_e, graph.edata.pop('a')
            else:
                return out_n, out_e


class KA_GAT(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, hidden_dim, out_1, out_2, gride_size, head, layer_num, pooling):
        super(KA_GAT, self).__init__()
        self.in_node_dim = in_node_dim
        self.in_edge_dim = in_edge_dim
        self.hidden_dim = hidden_dim
        self.out_1 = out_1
        self.out_2 = out_2
        self.head = head
        self.layer = layer_num

        self.grid_size = gride_size
        self.pooling = pooling

        self.node_kan_line = KAN_linear(in_node_dim, hidden_dim, gride_size, addbias=False)
        self.edge_kan_line = KAN_linear(in_edge_dim, hidden_dim, gride_size, addbias=False)

        self.attentions = nn.ModuleList()

        self.attentions.append(Gat_Kan_layer(in_node_feats=in_node_dim, in_edge_feats=in_edge_dim,
                                             out_node_feats=hidden_dim, out_edge_feats=hidden_dim,
                                             num_heads=self.head, grid_size=self.grid_size))

        for _ in range(self.layer - 1):
            self.attentions.append(Gat_Kan_layer(in_node_feats=hidden_dim, in_edge_feats=hidden_dim,
                                                 out_node_feats=hidden_dim, out_edge_feats=hidden_dim,
                                                 num_heads=self.head, grid_size=self.grid_size))
        self.leaky_relu = nn.LeakyReLU()
        out_layers = [
            KAN_linear(hidden_dim, out_1, gride_size, addbias=False),
            self.leaky_relu,
            KAN_linear(out_1, out_2, gride_size, addbias=True),
        ]
        self.Readout = nn.Sequential(*out_layers)

    def forward(self, g, node_feature, edge_feature):
        '''
        hidden_v = self.node_kan_line(node_feature)
        node_feature = F.leaky_relu(hidden_v)

        hidden_e = self.edge_kan_line(edge_feature)
        edge_feature = F.leaky_relu(hidden_e)
        '''
        for i in range(len(self.attentions)):
            atten = self.attentions[i]
            node_feature, edge_feature = atten(g, node_feature, edge_feature)

        out1 = F.leaky_relu(node_feature)
        out = self.Readout(out1)
        return out


# KAGNNModel
class KAGNNModel(nn.Module):
    """基于KA_GNN的药物分子特征提取器，输出原子级特征"""
    def __init__(self, node_input_dim=92,edge_input_dim=21,hidden_dim=128, output_dim=256,
                 grid_feat=8,num_head=2, num_layers=4, max_nodenum=268):
        super(KAGNNModel, self).__init__()
        self.max_nodenum = max_nodenum
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # KA_GNN_two用于提取原子级特征
        self.kagnn = KA_GAT(
            in_node_dim=node_input_dim,  # 输入特征维度（原子特征）
            in_edge_dim=edge_input_dim,  # 隐藏层维度
            hidden_dim=hidden_dim,  # 输出原子级特征维度
            out_1=output_dim,  # 输出维度
            out_2=output_dim,  # KAN网格大小
            gride_size=grid_feat,
            head=num_head,  # KAN层数
            layer_num=num_layers,
            pooling=True
        )

        # 投影层，用于进一步处理原子特征
        self.atom_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

    def forward(self, batched_graph):
        """
        Args:
            batched_graph: 批处理的DGL图
        Returns:
            atom_features_padded: 补零后的原子级特征 [batch_size, max_nodenum, output_dim]
        """
        # 1. 从批处理的DGL图中提取节点特征
        node_features = batched_graph.ndata['feat']  # [所有节点的总和, node_input_dim]
        edge_features = batched_graph.edata['feat']
        # 2. 通过KA_GNN_two获取原子级特征
        atom_features = self.kagnn(batched_graph, node_features,edge_features)  # [所有节点的总和, output_dim]

        # 4. 将批处理的图分解为单个图，以便进行补零操作
        batch_size = batched_graph.batch_size
        graphs = dgl.unbatch(batched_graph)

        # 5. 为每个图创建补零后的原子特征矩阵
        atom_features_padded = []

        for graph_idx, graph in enumerate(graphs):
            num_nodes = graph.num_nodes()

            # 获取当前图的原子特征
            start_idx = sum(g.num_nodes() for g in graphs[:graph_idx])
            end_idx = start_idx + num_nodes
            graph_atom_features = atom_features[start_idx:end_idx]  # [num_nodes, output_dim]

            # 如果节点数超过最大值，进行截断
            if num_nodes > self.max_nodenum:
                graph_atom_features = graph_atom_features[:self.max_nodenum]
                num_nodes = self.max_nodenum

            # 创建补零矩阵
            padded_features = torch.zeros((self.max_nodenum, self.output_dim),
                                          device=atom_features.device)
            padded_features[:num_nodes] = graph_atom_features[:num_nodes]

            atom_features_padded.append(padded_features.unsqueeze(0))

        # 6. 堆叠所有图的补零特征
        atom_features_padded = torch.cat(atom_features_padded, dim=0)  # [batch_size, max_nodenum, output_dim]
        return atom_features_padded


class LBSDTIAModel(nn.Module):
    """Model for predicting protein-drug binding affinity with GNN for drug representation"""
    """
    davis_data's max_numnode is 83,kiba_data's max_numnode is 194
    """
    def __init__(self, protein_emb=1024, gnn_hidden_dim=128, gnn_output_dim=256,
                 probability_dim=1500, hidden_dim=128, output_dim=256, max_nodenum=83):
        super(LBSDTIAModel, self).__init__()

        if probability_dim is None:
            raise ValueError("probability_dim must be specified")
        self.output_dim = output_dim
        self.protein_dim = protein_emb
        self.p_kernel_size = 3
        self.stride = 1
        self.dropout = nn.Dropout(0.3)
        self.drug_dim = gnn_output_dim  # 使用GNN输出的维度作为药物表征维度
        self.h_out = 4
        self.max_nodenum = max_nodenum
        # GNN for processing drug(KAN) - 输入直接是DGL图，输出补零后的原子级特征
        self.gnn = KAGNNModel(
            node_input_dim=92,  # 原子特征维度（根据您提供的数据，feat维度是92）
            edge_input_dim=21,
            hidden_dim=gnn_hidden_dim,  # 隐藏层维度
            output_dim=gnn_output_dim,  # 原子级输出维度
            grid_feat=3,  # KAN网格大小
            num_head=2,
            num_layers=2,  # KAN层数
            max_nodenum=max_nodenum  # 最大原子数
        )

        # Probability processing MLP
        self.probability_mlp = nn.Sequential(
            nn.Linear(probability_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, probability_dim),
            nn.ReLU(),
        )

        self.protein_mlp = nn.Sequential(
            nn.Linear(protein_emb, output_dim),
            nn.ReLU(),
        )

        # Final prediction MLP
        self.final_mlp = nn.Sequential(
            nn.Linear(self.output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(output_dim // 2, 1),
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(probability_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, probability_dim),
            nn.Sigmoid(),  # 输出门控信号（0~1）
        )

        # 轻量化注意力机制
        self.feature_conv = nn.Conv1d(self.protein_dim, self.protein_dim, self.p_kernel_size,
                                      stride=self.stride, padding=self.p_kernel_size // 2)
        self.attention_convolution = nn.Conv1d(self.protein_dim, self.protein_dim, self.p_kernel_size,
                                               stride=self.stride, padding=self.p_kernel_size // 2)

        # 双线性注意力机制
        self.bcn = weight_norm(
            BANLayer(v_dim=self.output_dim, q_dim=self.output_dim, h_dim=512, h_out=self.h_out),
            name='h_mat', dim=None)

    def forward(self, protein_emb, batched_graph, probability_vec):
        """
        Args:
            protein_emb: Tensor of shape (batch_size, seq_len, protein_emb)
            batched_graph: 批处理的DGL图，包含药物分子结构信息
            probability_vec: Tensor of shape (batch_size, probability_dim)
        """
        batch_size = protein_emb.shape[0]

        # 1. 处理药物DGL图：通过GNN获取补零后的原子级特征
        atom_features_padded = self.gnn(batched_graph)  # (batch_size, max_nodenum, gnn_output_dim)
        # print("2.", atom_features_padded.shape)
        # 5. 处理概率向量和蛋白质表征
        prob_weights = self.probability_mlp(probability_vec)  # (batch_size, probability_dim)
        # 6. 处理蛋白质表征
        protein_emb_lowdim = self.protein_mlp(protein_emb)  # (batch_size, seq_len, output_dim)
        # print("1.", protein_emb_lowdim.shape)
        # 7. 双线性注意力机制
        f, att = self.bcn(v=atom_features_padded, q=protein_emb_lowdim)
        # 8. 最终预测
        affinity = self.final_mlp(f)  # (batch_size, 1)
        return affinity.squeeze(-1)