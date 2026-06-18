import dgl
import torch
import numpy as np
import torch.nn as nn
import dgl.function as fn
from BanLayeratt import BANLayer
import torch.nn.functional as F
from rdkit import Chem, RDLogger
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

# KA-GCN
class KAN_linear(nn.Module):
    def __init__(self, inputdim, outdim, gridsize, addbias=True):
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
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        if self.addbias:
            y += self.bias
        y = y.view(outshape)
        return y


class NaiveFourierKANLayer(nn.Module):
    def __init__(self, in_feats, out_feats, gridsize, addbias=True):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.fouriercoeffs = nn.Parameter(torch.randn(2, out_feats, in_feats, gridsize) /
                                          (np.sqrt(in_feats) * np.sqrt(gridsize)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(out_feats))

    def forward(self, g, x):
        with g.local_scope():
            g.ndata['h'] = x

            g.update_all(message_func=self.fourier_transform, reduce_func=fn.sum(msg='m', out='h'))
            # If there is a bias, add it after message passing
            if self.addbias:
                g.ndata['h'] += self.bias
            return g.ndata['h']

    def fourier_transform(self, edges):
        src_feat = edges.src['h']  # Shape: (E, in_feats)

        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=src_feat.device), (1, 1, 1, self.gridsize))
        src_rshp = src_feat.view(src_feat.shape[0], 1, src_feat.shape[1], 1)
        cos_kx = torch.cos(k * src_rshp)
        sin_kx = torch.sin(k * src_rshp)

        # Reshape for multiplication
        cos_kx = torch.reshape(cos_kx, (1, src_feat.shape[0], src_feat.shape[1], self.gridsize))
        sin_kx = torch.reshape(sin_kx, (1, src_feat.shape[0], src_feat.shape[1], self.gridsize))

        # Perform Fourier transform using einsum
        m = torch.einsum("dbik,djik->bj", torch.concat([cos_kx, sin_kx], axis=0), self.fouriercoeffs)

        # Returning the message to be reduced
        return {'m': m}


class KA_GNN_two(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, use_bias=False):
        super(KA_GNN_two, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.kan_line = KAN_linear(in_feat, hidden_feat, grid_feat, addbias=use_bias)
        # self.linear_1 = KAN_linear(hidden_feat, out_feat, 1, addbias=True)
        # for _ in range(num_layers - 1):
        #     self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))
        # 逐层递增维度的KAN层
        # 第一层KANLayer（索引0）保持hidden_feat不变
        # 从第二层KANLayer（索引1）开始乘2倍
        for layer_idx in range(num_layers - 1):
            if layer_idx == 0:
                # 第一层KANLayer：输入输出都是hidden_feat
                layer_in_dim = hidden_feat
                layer_out_dim = hidden_feat
            else:
                # 后续层：输入是前一层输出，输出是前一层输出的2倍
                # 计算前一层的输出维度
                prev_out_dim = hidden_feat * (2 ** (layer_idx - 1))
                layer_in_dim = prev_out_dim
                layer_out_dim = prev_out_dim * 2

            self.layers.append(NaiveFourierKANLayer(
                layer_in_dim, layer_out_dim, grid_feat, addbias=use_bias
            ))
        # 计算最后一层的输出维度
        if num_layers - 1 <= 1:
            # 如果只有1层KANLayer，最后一层的输入维度是hidden_feat
            last_layer_input_dim = hidden_feat
        else:
            # 否则计算最后一层的输出维度
            last_layer_input_dim = hidden_feat * (2 ** (num_layers - 2))
        # 最后一层：将增加的维度映射回out_feat
        self.linear_1 = KAN_linear(last_layer_input_dim, 1024, 1, addbias=True)
        self.linear_2 = KAN_linear(1024, out_feat, 1, addbias=True)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_feat * (2 ** i)) for i in range(num_layers - 1)])

    def forward(self, g, h):
        h = self.kan_line(h)

        for i, layer in enumerate(self.layers):
            m = layer(g, h)
            m = self.norms[i](m)
            # h = nn.functional.leaky_relu(torch.add(m, h))
            h = nn.functional.leaky_relu(m)

        h = self.linear_1(h)
        h = self.linear_2(h)
        return h


# KAGNNModel
class KAGNNModel(nn.Module):
    # ... __init__ 保持不变 ...
    def __init__(self, node_input_dim=92, hidden_dim=128, output_dim=256,
                 grid_feat=8, num_layers=4, max_nodenum=268):
        super(KAGNNModel, self).__init__()
        self.max_nodenum = max_nodenum
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.kagnn = KA_GNN_two(
            in_feat=node_input_dim, hidden_feat=hidden_dim, out_feat=output_dim,
            out=output_dim, grid_feat=grid_feat, num_layers=num_layers, use_bias=True
        )

    def forward(self, batched_graph):
        """
        Returns:
            atom_features_padded: [batch_size, max_nodenum, output_dim]
            atom_masks: [batch_size, max_nodenum]  (新增返回项)
        """
        node_features = batched_graph.ndata['feat']
        batched_graph = update_node_features(batched_graph)
        atom_features = self.kagnn(batched_graph, node_features)

        graphs = dgl.unbatch(batched_graph)
        atom_features_padded = []
        atom_masks = []  # 新增：用于存储掩码

        for graph_idx, graph in enumerate(graphs):
            num_nodes = graph.num_nodes()
            start_idx = sum(g.num_nodes() for g in graphs[:graph_idx])
            end_idx = start_idx + num_nodes
            graph_atom_features = atom_features[start_idx:end_idx]

            if num_nodes > self.max_nodenum:
                graph_atom_features = graph_atom_features[:self.max_nodenum]
                num_nodes = self.max_nodenum

            # 特征补零
            padded_features = torch.zeros((self.max_nodenum, self.output_dim),
                                          device=atom_features.device)
            padded_features[:num_nodes] = graph_atom_features[:num_nodes]
            atom_features_padded.append(padded_features.unsqueeze(0))

            # 【新增】Mask 补零
            # 有原子的位置是 1，Padding 的位置是 0
            padded_mask = torch.zeros((self.max_nodenum), device=atom_features.device)
            padded_mask[:num_nodes] = 1.0
            atom_masks.append(padded_mask.unsqueeze(0))

        atom_features_padded = torch.cat(atom_features_padded, dim=0)
        atom_masks = torch.cat(atom_masks, dim=0)  # [batch_size, max_nodenum]

        return atom_features_padded, atom_masks


class LBSDTIAModel(nn.Module):
    """Model for predicting protein-drug binding affinity with GNN for drug representation"""
    """
    davis_data's max_numnode is 83,kiba_data's max_numnode is 194
    """
    def __init__(self, protein_emb=1024, gnn_hidden_dim=128, gnn_output_dim=256,
                 probability_dim=1500, hidden_dim=128, output_dim=256, max_nodenum=194):
        super(LBSDTIAModel, self).__init__()

        if probability_dim is None:
            raise ValueError("probability_dim must be specified")
        self.output_dim = output_dim
        self.protein_dim = protein_emb
        self.p_kernel_size = 3
        self.stride = 1
        self.gnn = KAGNNModel(
            node_input_dim=92, hidden_dim=gnn_hidden_dim, output_dim=gnn_output_dim,
            grid_feat=1, num_layers=3, max_nodenum=max_nodenum
        )
        self.protein_mlp = nn.Sequential(nn.Linear(protein_emb, output_dim), nn.ReLU())
        self.feature_conv = nn.Conv1d(self.protein_dim, self.protein_dim, self.p_kernel_size,
                                      stride=self.stride, padding=self.p_kernel_size // 2)
        self.attention_convolution = nn.Conv1d(self.protein_dim, self.protein_dim, self.p_kernel_size,
                                               stride=self.stride, padding=self.p_kernel_size // 2)
        self.softmax = nn.Softmax(dim=-1)
        self.bcn = weight_norm(
            BANLayer(v_dim=self.output_dim, q_dim=self.output_dim, h_dim=512, h_out=4),
            name='h_mat', dim=None)

        # Final prediction MLP (需补全定义以防漏掉)
        self.final_mlp = nn.Sequential(
            nn.Linear(self.output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(output_dim // 2, 1),
        )

    def forward(self, protein_emb, batched_graph, probability_vec, return_dual=False):
        """
        Args:
            protein_emb: [batch_size, seq_len, protein_dim] (含有 padding 的原始嵌入)
        """
        # 1. 获取药物特征和 Mask
        # atom_mask: [batch, max_nodenum]
        atom_features_padded, atom_mask = self.gnn(batched_graph)

        # 2. 生成蛋白质 Mask (在切片之前)
        # 假设 protein_emb 中全 0 的向量即为 padding
        # shape: [batch, seq_len] -> 1 for valid, 0 for pad
        protein_mask = (torch.sum(torch.abs(protein_emb), dim=-1) != 0).float()

        # 3. 处理蛋白质特征
        # 切片截断 (Mask 也要同步截断)
        protein_emb = protein_emb[:, :1000, :]
        protein_mask = protein_mask[:, :1000]  # [batch, 1000]

        # 卷积处理
        # 注意：Conv1d padding='same' (kernel//2)，所以长度保持不变，Mask 依然有效
        protein_emb = protein_emb.transpose(1, 2)  # (batch, dim, seq_len)
        protein_emb_0 = self.feature_conv(protein_emb)
        protein_emb_1 = self.attention_convolution(protein_emb)
        protein_emb = protein_emb_0 * self.softmax(protein_emb_1)

        protein_emb = protein_emb.transpose(1, 2)  # (batch, seq_len, dim)
        protein_emb_lowdim = self.protein_mlp(protein_emb)  # (batch, seq_len, output_dim)

        # 4. 双线性注意力(传入双向 Mask)
        # v: [B, V, D], q: [B, Q, D]
        # v_mask: [B, V], q_mask: [B, Q]
        f, att = self.bcn(
            v=atom_features_padded,
            q=protein_emb_lowdim,
            v_mask=atom_mask,
            q_mask=protein_mask
        )
        # 5. 最终预测
        affinity = self.final_mlp(f)

        if return_dual:
            noise_std = 0.05
            noise = torch.randn_like(f) * noise_std
            f1 = f + noise
            affinity1 = self.final_mlp(f1)
            return affinity.squeeze(-1), affinity1.squeeze(-1)
        else:
            return affinity.squeeze(-1)