import dgl
import torch
import numpy as np
import torch.nn as nn
import dgl.function as fn
from BanLayer import BANLayer
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

    def forward(self, g, h):
        h = self.kan_line(h)

        for i, layer in enumerate(self.layers):
            m = layer(g, h)
            # h = nn.functional.leaky_relu(torch.add(m, h))
            h = nn.functional.leaky_relu(m)

        h = self.linear_1(h)
        h = self.linear_2(h)
        return h


# KAGNNModel
class KAGNNModel(nn.Module):
    """基于KA_GNN的药物分子特征提取器，输出原子级特征"""

    def __init__(self, node_input_dim=92, hidden_dim=128, output_dim=256,
                 grid_feat=8, num_layers=4, max_nodenum=268):
        super(KAGNNModel, self).__init__()
        self.max_nodenum = max_nodenum
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # KA_GNN_two用于提取原子级特征
        self.kagnn = KA_GNN_two(
            in_feat=node_input_dim,  # 输入特征维度（原子特征）
            hidden_feat=hidden_dim,  # 隐藏层维度
            out_feat=output_dim,  # 输出原子级特征维度
            out=output_dim,  # 输出维度
            grid_feat=grid_feat,  # KAN网格大小
            num_layers=num_layers,  # KAN层数
            use_bias=True
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
        batched_graph = update_node_features(batched_graph)
        # 2. 通过KA_GNN_two获取原子级特征
        atom_features = self.kagnn(batched_graph, node_features)  # [所有节点的总和, output_dim]

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
        self.softmax = nn.Softmax(dim=-1)
        self.top_k_sites = None
        self.binding_site_threshold = 0.8
        self.max_binding_sites = 1500

        # GNN for processing drug(KAN) - 输入直接是DGL图，输出补零后的原子级特征
        self.gnn = KAGNNModel(
            node_input_dim=92,  # 原子特征维度（根据您提供的数据，feat维度是92）
            hidden_dim=gnn_hidden_dim,  # 隐藏层维度
            output_dim=gnn_output_dim,  # 原子级输出维度
            grid_feat=1,  # KAN网格大小
            num_layers=4,  # KAN层数
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
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(output_dim // 2, 1),
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
        self.bcn1 = weight_norm(
            BANLayer(v_dim=self.output_dim, q_dim=self.output_dim, h_dim=512, h_out=self.h_out),
            name='h_mat', dim=None)

    def select_and_pad_binding_sites(self, protein_lowdim, probability_vec):
        """
        根据概率向量选择结合位点并进行填充对齐
        Args:
            protein_lowdim: (batch_size, seq_len, output_dim) 已映射到低维的蛋白质特征
            probability_vec: (batch_size, probability_dim) 结合概率向量
        Returns:
            binding_sites_padded: (batch_size, max_binding_sites, output_dim) 填充后的结合位点特征
            mask: (batch_size, max_binding_sites) 有效位置掩码
        """
        batch_size, seq_len, output_dim = protein_lowdim.shape

        # 确保probability_vec的维度与蛋白质序列长度匹配
        if probability_vec.size(1) != seq_len:
            # 调整概率向量维度到序列长度
            probability_vec = F.interpolate(
                probability_vec.unsqueeze(1),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).squeeze(1)

        binding_sites_padded = []
        valid_masks = []
        all_indices = []

        for i in range(batch_size):
            # 获取当前样本的概率向量
            prob = probability_vec[i]  # (seq_len,)

            # 确定结合位点索引
            if self.top_k_sites is not None:
                # 选取概率最高的top_k个位点
                top_k = min(self.top_k_sites, seq_len)
                _, indices = torch.topk(prob, k=top_k)  # (top_k,)
            else:
                # 使用阈值确定结合位点
                indices = torch.where(prob > self.binding_site_threshold)[0]  # (num_sites,)

                # 如果没有检测到结合位点，使用概率最高的3个位置
                if len(indices) == 0:
                    _, indices = torch.topk(prob, k=min(300, seq_len))

            all_indices.append(indices)  # (batch_size, num_sites)
            # 提取结合位点特征
            binding_sites = protein_lowdim[i, indices]  # (num_sites, output_dim)
            num_sites = binding_sites.size(0)

            # 创建填充矩阵
            padded_sites = torch.zeros((self.max_binding_sites, output_dim),
                                       device=protein_lowdim.device)

            # 创建有效位置掩码
            valid_mask = torch.zeros(self.max_binding_sites,
                                     device=protein_lowdim.device,
                                     dtype=torch.bool)

            # 填充有效结合位点
            actual_sites = min(num_sites, self.max_binding_sites)
            padded_sites[:actual_sites] = binding_sites[:actual_sites]
            valid_mask[:actual_sites] = True

            binding_sites_padded.append(padded_sites.unsqueeze(0))
            valid_masks.append(valid_mask.unsqueeze(0))

        # 堆叠所有样本
        binding_sites_padded = torch.cat(binding_sites_padded, dim=0)  # (batch_size, max_binding_sites, output_dim)
        valid_masks = torch.cat(valid_masks, dim=0)  # (batch_size, max_binding_sites)

        return binding_sites_padded, valid_masks, all_indices


    def forward(self, protein_emb, batched_graph, probability_vec, return_dual=False):
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
        # 2. 处理概率向量和蛋白质表征
        # # 3. 处理蛋白质表征
        protein_emb = protein_emb[:, :1000, :]
        protein_emb = protein_emb.transpose(1, 2)  # (batch_size, protein_dim， seq_len)
        protein_emb_0 = self.feature_conv(protein_emb)
        protein_emb_1 = self.attention_convolution(protein_emb)
        protein_emb = protein_emb_0 * self.softmax(protein_emb_1)
        protein_emb = protein_emb.transpose(1, 2)  # (batch_size, seq_len, protein_dim)
        protein_emb_lowdim = self.protein_mlp(protein_emb)  # (batch_size, seq_len, output_dim)

        binding_sites_padded, valid_masks, all_indices = self.select_and_pad_binding_sites(
            protein_emb_lowdim, probability_vec
        )  # (batch_size, max_binding_sites, output_dim)
        # print("1.", all_indices)

        # 4. 双线性注意力机制
        f, att = self.bcn(v=atom_features_padded,
                          q=protein_emb_lowdim)  # 全序列(batch_size, head, max_drug_node,max_binding_sites)
        # print("1,", att.shape)
        f1, att1 = self.bcn1(v=atom_features_padded,
                            q=binding_sites_padded)  # 结合位点序列(batch_size, head, max_drug_node,max_binding_sites)
        # print("2,", att1.shape)
        # print("3,", binding_sites_padded.shape)
        att_sum = torch.sum(torch.sum(att, dim=1), dim=1)
        layer_norm1 = nn.LayerNorm(att_sum.size(1), elementwise_affine=False)
        n_att_sum = layer_norm1(att_sum)  # (batch_size, 1500)
        att1_sum = torch.sum(torch.sum(att1, dim=1), dim=1)
        layer_norm2 = nn.LayerNorm(att1_sum.size(1), elementwise_affine=False)
        n_att1_sum = layer_norm2(att1_sum)  # (batch_size, 1000)
        selected_att_sums = []  # 存放选择结果的列表
        # print("4.", n_att_sum.shape)
        # print("5,", n_att1_sum.shape)
        for batch_idx, indices in enumerate(all_indices):
            # 确保索引在有效范围内
            valid_indices = indices[indices < n_att_sum.shape[1]]
            # 从n_att_sum中选择对应位置的值
            selected_values = n_att_sum[batch_idx, valid_indices]
            # 将结果添加到列表中
            selected_att_sums.append(selected_values)
        selected_att1_sums = []
        for batch_idx, indices in enumerate(all_indices):
            # indices: 当前batch的结合位点索引
            # n_att1_sum[batch_idx]: 当前batch的归一化注意力求和 (300,)
            # 获取当前batch的结合位点数量
            num_sites = len(indices)
            # 确保不超过n_att1_sum的维度
            num_to_select = min(num_sites, n_att1_sum.shape[1])
            # 从n_att1_sum中选择前num_sites个值
            if num_to_select > 0:
                selected_values = n_att1_sum[batch_idx, :num_to_select]
            else:
                # 如果没有结合位点，选择空Tensor
                selected_values = torch.tensor([], device=n_att1_sum.device)
            # 将结果添加到列表中
            selected_att1_sums.append(selected_values)

        size_match = True
        for batch_idx, (att_tensor, att1_tensor) in enumerate(zip(selected_att_sums, selected_att1_sums)):
            if att_tensor.shape != att1_tensor.shape:
                print(f"Batch {batch_idx}: 大小不匹配!")
                print(f"  selected_att_sums[{batch_idx}].shape = {att_tensor.shape}")
                print(f"  selected_att1_sums[{batch_idx}].shape = {att1_tensor.shape}")
                size_match = False

        # 5. 最终预测
        affinity = self.final_mlp(f)  # (batch_size, 1)
        # 6.加噪并输出结果
        if return_dual:
            noise_std = 0.05
            noise = torch.randn_like(f) * noise_std
            f1 = f + noise
            affinity1 = self.final_mlp(f1)
            return affinity.squeeze(-1), affinity1.squeeze(-1), selected_att_sums, selected_att1_sums
        else:
            return affinity.squeeze(-1), selected_att_sums, selected_att1_sums