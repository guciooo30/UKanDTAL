import dgl
import torch
import BanLayer
import numpy as np
import torch.nn as nn
import dgl.function as fn
from torch.xpu import device
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch.ao.nn.quantized import Dropout
from torch.nn import Sequential, Linear, ReLU
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool as gap, global_max_pool as gmp
from dgl.nn import SortPooling, WeightAndSum, GlobalAttentionPooling, Set2Set, SumPooling, AvgPooling, MaxPooling
RDLogger.DisableLog('rdApp.*')

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    """DeepDTA版本的原子特征，使用正确的新API"""
    # 完全修复：使用新的GetValence API
    try:
        # 新版本的RDKit API - 明确指定getExplicit参数
        implicit_valence = atom.GetImplicitValence()
    except Exception as e:
        # 如果API调用有问题，使用默认值
        print(f"Warning: GetImplicitValence failed: {e}, using default value 0")
        implicit_valence = 0

    features = np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
            'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
            'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
            'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
        ]) +  # 44维
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +  # 11维
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +  # 11维
        one_of_k_encoding_unk(implicit_valence, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +  # 11维
        [atom.GetIsAromatic()]  # 1维
    )

    return features

# davis的最大原子个数为46，kiba为268
def smiles_to_graph(smiles):
    """使用DeepDTA的原子特征"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None, None

    # ========== DeepDTA原子特征 ==========
    atom_features_list = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        # 特征归一化（与DeepDTA一致）
        feature_sum = sum(feature)
        if feature_sum > 0:
            normalized_feature = feature / feature_sum
        else:
            normalized_feature = feature
        atom_features_list.append(normalized_feature)

    # 修复性能警告：先转换为numpy数组再转tensor
    atom_features_array = np.array(atom_features_list)
    node_feats = torch.tensor(atom_features_array, dtype=torch.float32)

    # ========== 边和边特征 ==========
    src, dst = [], []
    edge_features = []
    for bond in mol.GetBonds():
        src.append(bond.GetBeginAtomIdx())
        dst.append(bond.GetEndAtomIdx())

        # 键特征
        bond_features = [
            float(bond.GetBondTypeAsDouble()),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            float(bond.GetStereo()),
        ]
        edge_features.append(bond_features)

    # 创建边索引 (无向图，需要添加反向边)
    if src and dst:
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    else:
        # 如果没有边，创建空的边索引
        edge_index = torch.tensor([[], []], dtype=torch.long)

    # 处理边特征
    if edge_features:
        # 修复性能警告
        edge_features_array = np.array(edge_features * 2)
        edge_feats = torch.tensor(edge_features_array, dtype=torch.float32)
    else:
        edge_feats = None

    return node_feats, edge_index, edge_feats

# 不使用边特征的GCN
class GraphConv(nn.Module):
    """简单的图卷积层实现"""

    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, edge_index):
        # x: [num_nodes, input_dim]
        # edge_index: [2, num_edges]
        row, col = edge_index
        # 聚合邻居信息 (简单平均)
        neighbor_agg = scatter_mean(x[row], col, dim=0, dim_size=x.size(0))
        # 线性变换并加上邻居聚合
        x = self.linear(x + neighbor_agg)
        return F.relu(x)

# 标准的GCN
class GCNLayer(nn.Module):
    """标准的图卷积网络层"""

    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):
        # x: [num_nodes, in_features]
        # edge_index: [2, num_edges]

        row, col = edge_index

        # 计算度矩阵（用于归一化）
        deg = scatter_add(torch.ones_like(row), row, dim=0, dim_size=x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # 对称归一化
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 消息传递和聚合
        messages = x[col] * norm.view(-1, 1)
        aggregated = scatter_add(messages, row, dim=0, dim_size=x.size(0))

        # 线性变换
        out = self.linear(aggregated)
        return F.relu(out)

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
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(KA_GNN_two, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        # self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.layers = nn.ModuleList()

        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.kan_line = KAN_linear(in_feat, hidden_feat, grid_feat, addbias=use_bias)

        for _ in range(num_layers - 1):
            self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))

        # self.layers.append()
        # self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))

        # self.layers.append(KAN_linear(hidden_feat, out_feat, grid_feat, addbias=use_bias))
        # self.layers.append(NaiveFourierKANLayer(hidden_feat, out_feat, grid_feat, addbias=use_bias))

        # self.layers.append(NaiveFourierKANLayer(out_feat, out_feat, grid_feat, addbias=use_bias))
        self.linear_1 = KAN_linear(hidden_feat, out, 1, addbias=True)
        # self.linear_2 = KAN_linear(out_feat, out, grid_feat, addbias=True)
        self.sumpool = SumPooling()
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

        layers_kan = [
            # nn.Linear(self.hidden_size*2, self.hidden_size),
            self.linear_1,
            nn.Sigmoid()
        ]

        self.Readout = nn.Sequential(*layers_kan)

    def forward(self, g, h):
        h = self.kan_line(h)

        for i, layer in enumerate(self.layers):
            m = layer(g, h)
            h = nn.functional.leaky_relu(torch.add(m, h))

        if self.pooling == 'avg':
            y = self.avgpool(g, h)

        elif self.pooling == 'max':
            y = self.maxpool(g, h)


        elif self.pooling == 'sum':
            y = self.sumpool(g, h)


        else:
            print('No pooling found!!!!')

        out = self.Readout(y)
        return out

    def get_grad_norm_weights(self) -> nn.Module:
        return self.parameters()

class GNNModel(nn.Module):
    def __init__(self, node_input_dim=78, edge_input_dim=4, hidden_dim=128, output_dim=256):
        super(GNNModel, self).__init__()
        self.node_embed = nn.Linear(node_input_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_input_dim, hidden_dim)
        self.gnn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gnn2 = GCNLayer(hidden_dim, hidden_dim * 2)
        self.gnn3 = GCNLayer(hidden_dim * 2, hidden_dim * 4)
        self.Dropout = nn.Dropout(p=0.3)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Dropout(p=0.2),
        )

    def forward(self, node_feats, edge_index, edge_feats=None):
        # 嵌入节点特征
        node_feats = self.node_embed(node_feats)

        # 如果有边特征，处理边特征
        if edge_feats is not None:
            edge_feats = self.edge_embed(edge_feats)
            # 在简单实现中，我们暂时不直接使用边特征影响消息传递
        # 图卷积
        node_feats = self.gnn1(node_feats, edge_index)
        node_feats = self.gnn2(node_feats, edge_index)
        node_feats = self.gnn3(node_feats, edge_index)
        # 读取图嵌入 (对所有节点特征取平均)(全局平均池化)
        # graph_embed = torch.mean(node_feats, dim=0, keepdim=True)
        # 对所有接待你特征取最大值（全局最大池化）
        graph_embed = torch.max(node_feats, dim=0, keepdim=True)[0]
        # 通过readout网络处理分子级图特征
        node_readout = self.readout(graph_embed)  # [num_nodes, output_dim]
        return node_readout

class LBSDTIAModel(nn.Module):
    """Model for predicting protein-drug binding affinity with GNN for drug representation"""

    def __init__(self, protein_emb=1024, gnn_hidden_dim=128, gnn_output_dim=256,
                 probability_dim=1500, hidden_dim=128, output_dim=256):
        super(LBSDTIAModel, self).__init__()

        if probability_dim is None:
            raise ValueError("probability_dim must be specified")
        self.output_dim = output_dim
        self.protein_dim = protein_emb
        self.p_kernel_size = 3
        self.stride = 1
        self.dropout = nn.Dropout(0.3)
        self.drug_dim = gnn_output_dim  # 使用GNN输出的维度作为药物表征维度
        self.softmax = nn.Softmax(dim=-1)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        # GNN for processing drug
        self.gnn = GNNModel(
            node_input_dim=78,  # 从smiles_to_graph函数中原子特征的维度
            edge_input_dim=4,  # 从smiles_to_graph函数中边特征的维度
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim
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
            nn.Linear(self.output_dim + self.drug_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(output_dim, output_dim//2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(output_dim//2, 1),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(probability_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, probability_dim),
            nn.Sigmoid(),  # 输出门控信号（0~1）
        )
        self.feature_conv = nn.Conv1d(self.protein_dim,self.protein_dim,self.p_kernel_size,
                                      stride=self.stride,padding=self.p_kernel_size // 2)
        self.attention_convolution = nn.Conv1d(self.protein_dim,self.protein_dim,self.p_kernel_size,
                                      stride=self.stride,padding=self.p_kernel_size // 2)

    def forward(self, protein_emb, drug_smiles_list, probability_vec):
        """
        Args:
            protein_emb: Tensor of shape (batch_size, seq_len, protein_emb)
            drug_smiles_list: List of SMILES strings of length batch_size
            probability_vec: Tensor of shape (batch_size,1, probability_dim(seq_len))
            note: probability_dim = seq_len
        """
        batch_size = len(drug_smiles_list)

        # 1. 处理药物SMILES：转换为图并获取GNN嵌入
        drug_embs = []
        for i in range(batch_size):
            # 将SMILES转换为图
            node_feats, edge_index, edge_feats = smiles_to_graph(drug_smiles_list[i])
            # print(node_feats.shape)
            # 如果图转换失败，使用零向量作为后备
            if node_feats is None:
                drug_emb = torch.zeros((1, self.drug_dim), device=protein_emb.device)
            else:
                # 将图数据移动到与protein_emb相同的设备
                node_feats = node_feats.to(protein_emb.device)
                edge_index = edge_index.to(protein_emb.device)
                if edge_feats is not None:
                    edge_feats = edge_feats.to(protein_emb.device)

                # 通过GNN获取药物嵌入
                drug_emb = self.gnn(node_feats, edge_index, edge_feats)

            drug_embs.append(drug_emb)

        # 堆叠所有药物嵌入
        drug_emb = torch.cat(drug_embs, dim=0)  # (batch_size, drug_dim)
        # print("0", drug_emb.shape)
        # 2. 处理概率向量和蛋白质表征
        prob_weights = self.probability_mlp(probability_vec)  # (batch_size, probability_dim)
        # print("1", prob_weights.shape)
        prob_weights = prob_weights.unsqueeze(-1)   # (batch_size, probability_dim, 1)
        # print("2", prob_weights.shape)
        prob_weights_expanded = prob_weights.expand(-1, -1, self.output_dim)  # (batch_size, probability_dim, 256)
        # print("3", protein_emb.shape)
        protein_emb_lowdim = self.protein_mlp(protein_emb)  # (batch_size, probability_dim, 256)
        # print("4", prob_weights_expanded.shape)
        # print("5", protein_emb.shape)
        # 3. 加权蛋白质表征
        weighted_protein = protein_emb_lowdim * prob_weights_expanded  # (batch_size, seq_len, protein_emb)
        # # 不加结合位点的信息
        # weighted_protein = protein_emb
        # 4. 沿序列长度维度求和
        prob_protein = weighted_protein.sum(dim=1)  # (batch_size, protein_emb)

        # # 5. 使用LA处理蛋白质表征
        # protein_emb_convert = protein_emb.permute(0, 2, 1)  # [batch, seq_len, dim] -> [batch, dim, seq_len]
        # protein_emb_convert = protein_emb_convert.to(device=protein_emb.device, dtype=torch.float32)
        # protein_feature = self.feature_conv(protein_emb_convert)
        # protein_feature = self.dropout(protein_feature)
        # attention = self.attention_convolution(protein_feature)
        # la_protein = torch.sum(protein_feature * self.softmax(attention), dim=-1)  # 加权平均池化  # [batch, dim]

        # # 6. 拼接蛋白质表征，并降维
        # concat_protein = torch.cat((la_protein, prob_protein), dim=1)
        # prob_protein = self.protein_mlp(prob_protein)

        # 7. 拼接蛋白质和药物表征
        combined = torch.cat([prob_protein, drug_emb], dim=1)  # (batch_size, protein_emb + drug_dim)
        # 8. 最终预测
        affinity = self.final_mlp(combined)  # (batch_size, 1)

        return affinity.squeeze(-1)

# import torch
# import torch.nn as nn
# import numpy as np
# import torch.nn.functional as F
# from torch.ao.nn.quantized import Dropout
# from torch_scatter import scatter_mean, scatter_add
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from dgl.nn import SortPooling, WeightAndSum, GlobalAttentionPooling, Set2Set, SumPooling, AvgPooling, MaxPooling
# import dgl
# import dgl.function as fn
#
#
# def smiles_to_dgl_graph(smiles, device):
#     """
#     将SMILES字符串转换为DGL图，使用详细的原子特征
#     """
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None, None
#
#     # 添加氢原子
#     mol = Chem.AddHs(mol)
#
#     # 创建DGL图
#     g = dgl.graph(([], []), idtype=torch.int32, device=device)
#
#     # 添加节点
#     num_atoms = mol.GetNumAtoms()
#     g.add_nodes(num_atoms)
#
#     # ========== 详细的原子特征 ==========
#     atom_features = []
#
#     # 计算3D坐标和Gasteiger电荷
#     try:
#         AllChem.EmbedMolecule(mol, randomSeed=42)
#         AllChem.ComputeGasteigerCharges(mol)
#
#         for atom in mol.GetAtoms():
#             features = encode_atom(atom)
#             atom_features.append(features)
#
#     except Exception as e:
#         # 如果3D坐标生成失败，仅基于2D信息
#         for atom in mol.GetAtoms():
#             features = encode_atom(atom)
#             atom_features.append(features)
#
#     node_feats = torch.tensor(atom_features, dtype=torch.float32, device=device)
#
#     # ========== 添加边 ==========
#     src_list, dst_list = [], []
#
#     # 添加化学键边
#     for bond in mol.GetBonds():
#         u = bond.GetBeginAtomIdx()
#         v = bond.GetEndAtomIdx()
#
#         src_list.extend([u, v])
#         dst_list.extend([v, u])
#
#     if src_list:
#         src_tensor = torch.tensor(src_list, dtype=torch.int32, device=device)
#         dst_tensor = torch.tensor(dst_list, dtype=torch.int32, device=device)
#         g.add_edges(src_tensor, dst_tensor)
#
#     # 添加节点特征
#     g.ndata['h'] = node_feats
#
#     return g, node_feats
#
#
# def encode_chirality(atom):
#     """编码手性信息"""
#     chirality_tags = [0] * 4
#
#     if atom.HasProp("_CIPCode"):
#         chirality = atom.GetProp("_CIPCode")
#         if chirality == "R":
#             chirality_tags[0] = 1
#         elif chirality == "S":
#             chirality_tags[1] = 1
#         elif chirality == "E":
#             chirality_tags[2] = 1
#         elif chirality == "Z":
#             chirality_tags[3] = 1
#
#     return chirality_tags
#
#
# def encode_atom(atom):
#     """编码原子特征，总维度47"""
#     # 芳香性: 2维
#     aromaticity = [0, 0]
#     aromaticity[int(atom.GetIsAromatic())] = 1
#
#     # 形式电荷: 16维 (-8到+7)
#     formal_charge = [0] * 16
#     formal_charge[atom.GetFormalCharge() + 8] = 1
#
#     # 手性标签: 4维
#     chirality_tags = encode_chirality(atom)
#
#     # 原子度数: 11维 (0-10)
#     degree = [0] * 11
#     actual_degree = min(atom.GetDegree(), 10)
#     degree[actual_degree] = 1
#
#     # 氢原子数: 9维 (0-8)
#     num_hydrogens = [0] * 9
#     actual_hydrogens = min(atom.GetTotalNumHs(), 8)
#     num_hydrogens[actual_hydrogens] = 1
#
#     # 杂化方式: 5维
#     hybridization = [0] * 5
#     hybridization_type = atom.GetHybridization()
#     hybridization_mapping = {
#         Chem.HybridizationType.SP: 0,
#         Chem.HybridizationType.SP2: 1,
#         Chem.HybridizationType.SP3: 2,
#         Chem.HybridizationType.SP3D: 3,
#         Chem.HybridizationType.SP3D2: 4
#     }
#     if hybridization_type in hybridization_mapping:
#         hybridization[hybridization_mapping[hybridization_type]] = 1
#
#     # 组合所有特征 (2+16+4+11+9+5=47维)
#     return aromaticity + formal_charge + chirality_tags + degree + num_hydrogens + hybridization
#
#
# # 保留KAN相关类
# class KAN_linear(nn.Module):
#     def __init__(self, inputdim, outdim, gridsize, addbias=True):
#         super(KAN_linear, self).__init__()
#         self.gridsize = gridsize
#         self.addbias = addbias
#         self.inputdim = inputdim
#         self.outdim = outdim
#
#         self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
#                                           (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
#         if self.addbias:
#             self.bias = nn.Parameter(torch.zeros(1, outdim))
#
#     def forward(self, x):
#         xshp = x.shape
#         outshape = xshp[0:-1] + (self.outdim,)
#         x = x.view(-1, self.inputdim)
#         k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
#         xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
#         c = torch.cos(k * xrshp)
#         s = torch.sin(k * xrshp)
#
#         c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
#         s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
#         y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
#         if self.addbias:
#             y += self.bias
#         y = y.view(outshape)
#         return y
#
#
# class NaiveFourierKANLayer(nn.Module):
#     def __init__(self, in_feats, out_feats, gridsize, addbias=True):
#         super(NaiveFourierKANLayer, self).__init__()
#         self.gridsize = gridsize
#         self.addbias = addbias
#         self.in_feats = in_feats
#         self.out_feats = out_feats
#
#         self.fouriercoeffs = nn.Parameter(torch.randn(2, out_feats, in_feats, gridsize) /
#                                           (np.sqrt(in_feats) * np.sqrt(gridsize)))
#         if self.addbias:
#             self.bias = nn.Parameter(torch.zeros(out_feats))
#
#     def forward(self, g, x):
#         with g.local_scope():
#             g.ndata['h'] = x
#             g.update_all(message_func=self.fourier_transform, reduce_func=fn.sum(msg='m', out='h'))
#             if self.addbias:
#                 g.ndata['h'] += self.bias
#             return g.ndata['h']
#
#     def fourier_transform(self, edges):
#         src_feat = edges.src['h']
#         k = torch.reshape(torch.arange(1, self.gridsize + 1, device=src_feat.device), (1, 1, 1, self.gridsize))
#         src_rshp = src_feat.view(src_feat.shape[0], 1, src_feat.shape[1], 1)
#         cos_kx = torch.cos(k * src_rshp)
#         sin_kx = torch.sin(k * src_rshp)
#
#         cos_kx = torch.reshape(cos_kx, (1, src_feat.shape[0], src_feat.shape[1], self.gridsize))
#         sin_kx = torch.reshape(sin_kx, (1, src_feat.shape[0], src_feat.shape[1], self.gridsize))
#
#         m = torch.einsum("dbik,djik->bj", torch.concat([cos_kx, sin_kx], axis=0), self.fouriercoeffs)
#         return {'m': m}
#
# class KA_GNN_two(nn.Module):
#     def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
#         super(KA_GNN_two, self).__init__()
#         self.num_layers = num_layers
#         self.pooling = pooling
#         self.layers = nn.ModuleList()
#
#         self.leaky_relu = nn.LeakyReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.kan_line = KAN_linear(in_feat, hidden_feat, grid_feat, addbias=use_bias)
#
#         for _ in range(num_layers - 1):
#             self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))
#
#         self.linear_1 = KAN_linear(hidden_feat, out, 1, addbias=True)
#         self.sumpool = SumPooling()
#         self.avgpool = AvgPooling()
#         self.maxpool = MaxPooling()
#
#         layers_kan = [
#             self.linear_1,
#             nn.Sigmoid()
#         ]
#
#         self.Readout = nn.Sequential(*layers_kan)
#
#     def forward(self, g, h):
#         h = self.kan_line(h)
#
#         for i, layer in enumerate(self.layers):
#             m = layer(g, h)
#             h = nn.functional.leaky_relu(torch.add(m, h))
#
#         if self.pooling == 'avg':
#             y = self.avgpool(g, h)
#         elif self.pooling == 'max':
#             y = self.maxpool(g, h)
#         elif self.pooling == 'sum':
#             y = self.sumpool(g, h)
#         else:
#             print('No pooling found!!!!')
#
#         out = self.Readout(y)
#         return out
#
#     def get_grad_norm_weights(self) -> nn.Module:
#         return self.parameters()
#
# class KA_GNN(nn.Module):
#     def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
#         super(KA_GNN, self).__init__()
#         self.num_layers = num_layers
#         self.pooling = pooling
#         self.kan_line = KAN_linear(in_feat, hidden_feat, grid_feat, addbias=use_bias)
#         self.layers = nn.ModuleList()
#         self.leaky_relu = nn.LeakyReLU()
#
#         for _ in range(num_layers - 1):
#             self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))
#
#         self.linear_1 = KAN_linear(hidden_feat, out_feat, grid_feat, addbias=use_bias)
#         self.linear_2 = KAN_linear(out_feat, out, grid_feat, addbias=use_bias)
#
#         self.sumpool = SumPooling()
#         self.avgpool = AvgPooling()
#         self.maxpool = MaxPooling()
#
#         self.Readout = nn.Sequential(
#             self.linear_1,
#             self.leaky_relu,
#             self.linear_2,
#             nn.Sigmoid()
#         )
#
#     def forward(self, g, features):
#         h = self.kan_line(features)
#         for layer in self.layers:
#             h = layer(g, h)
#
#         if self.pooling == 'avg':
#             y = self.avgpool(g, h)
#         elif self.pooling == 'max':
#             y = self.maxpool(g, h)
#         elif self.pooling == 'sum':
#             y = self.sumpool(g, h)
#         else:
#             raise ValueError(f"Unknown pooling type: {self.pooling}")
#
#         return self.Readout(y)
#
#
# class LBSDTIAModel(nn.Module):
#     """Model for predicting protein-drug binding affinity with KA_GNN for drug representation"""
#
#     def __init__(self, protein_emb=1024, drug_emb_dim=256,
#                  probability_dim=1500, hidden_dim=128, output_dim=256,
#                  kan_hidden_dim=128, kan_grid_size=8, kan_num_layers=3, kan_pooling='sum'):
#         super(LBSDTIAModel, self).__init__()
#
#         if probability_dim is None:
#             raise ValueError("probability_dim must be specified")
#
#         # KA_GNN for processing drug - 使用47维原子特征
#         self.ka_gnn = KA_GNN_two(
#             in_feat=47,
#             hidden_feat=kan_hidden_dim,
#             out_feat=kan_hidden_dim * 2,
#             out=drug_emb_dim,
#             grid_feat=kan_grid_size,
#             num_layers=kan_num_layers,
#             pooling=kan_pooling,
#             use_bias=True
#         )
#
#         # Probability processing MLP
#         self.probability_mlp = nn.Sequential(
#             nn.Linear(probability_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, probability_dim),
#             nn.ReLU(),
#         )
#
#         self.protein_mlp = nn.Sequential(
#             nn.Linear(protein_emb, output_dim),
#             nn.ReLU(),
#         )
#
#         self.output_dim = output_dim
#         self.drug_dim = drug_emb_dim
#
#         # Final prediction MLP
#         self.final_mlp = nn.Sequential(
#             nn.Linear(self.output_dim + self.drug_dim, output_dim),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(output_dim, output_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(output_dim // 2, 1),
#         )
#
#     def forward(self, protein_emb, drug_smiles_list, probability_vec):
#         batch_size = len(drug_smiles_list)
#         device = protein_emb.device
#
#         probability_vec = probability_vec.to(device)
#
#         # 1. 处理药物SMILES
#         drug_embs = []
#         for i in range(batch_size):
#             g, node_feats = smiles_to_dgl_graph(drug_smiles_list[i], device)
#
#             if g is None or node_feats is None:
#                 drug_emb = torch.zeros((1, self.drug_dim), device=device)
#             else:
#                 try:
#                     drug_emb = self.ka_gnn(g, node_feats)
#                 except Exception as e:
#                     print(f"Error processing molecule {i}: {drug_smiles_list[i]}")
#                     print(f"Error: {e}")
#                     drug_emb = torch.zeros((1, self.drug_dim), device=device)
#
#             drug_embs.append(drug_emb)
#
#         drug_emb = torch.cat(drug_embs, dim=0)
#
#         # 2. 处理概率向量和蛋白质表征
#         prob_weights = self.probability_mlp(probability_vec)
#         prob_weights = prob_weights.unsqueeze(-1)
#         prob_weights_expanded = prob_weights.expand(-1, -1, self.output_dim)
#
#         protein_emb = self.protein_mlp(protein_emb)
#
#         # 3. 加权蛋白质表征
#         weighted_protein = protein_emb * prob_weights_expanded
#         pooled_protein = weighted_protein.sum(dim=1)
#
#         # 4. 拼接蛋白质和药物表征
#         combined = torch.cat([pooled_protein, drug_emb], dim=1)
#
#         # 5. 最终预测
#         affinity = self.final_mlp(combined)
#
#         return affinity.squeeze(-1)