import dgl
import torch
import numpy as np
import torch.nn as nn
import dgl.function as fn
from torch.xpu import device
from BanLayer import BANLayer
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch.ao.nn.quantized import Dropout
from torch.nn import Sequential, Linear, ReLU
from torch.nn.utils.weight_norm import weight_norm
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
def smiles_to_graph(smiles, max_nodenum=46):
    """
    将SMILES转换为图数据结构，并将节点特征补零到固定维度
    Args:
        smiles: SMILES字符串
        max_nodenum: 最大节点数，默认为100
    Returns:
        node_feats: 补零后的节点特征 [max_nodenum, feature_dim]
        edge_index: 边索引 [2, num_edges]
        edge_feats: 边特征 [num_edges, edge_feat_dim] 或 None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

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

    # 转换为numpy数组
    atom_features_array = np.array(atom_features_list)
    num_nodes = len(atom_features_array)
    feature_dim = atom_features_array.shape[1]

    # ========== 创建补零后的节点特征矩阵 ==========
    # 如果节点数超过最大值，进行截断
    if num_nodes > max_nodenum:
        # 截断到最大节点数
        atom_features_array = atom_features_array[:max_nodenum]
        num_nodes = max_nodenum

    # 创建补零矩阵
    node_feats_padded = np.zeros((max_nodenum, feature_dim), dtype=np.float32)
    node_feats_padded[:num_nodes] = atom_features_array

    # 转换为tensor
    node_feats = torch.tensor(node_feats_padded, dtype=torch.float32)

    # ========== 边和边特征 ==========
    src, dst = [], []
    edge_features = []

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        # 如果节点索引超过了最大节点数，跳过这条边
        if begin_idx >= max_nodenum or end_idx >= max_nodenum:
            continue

        src.append(begin_idx)
        dst.append(end_idx)

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
        self.readout1 = nn.Sequential(
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
        # # 读取图嵌入 (对所有节点特征取平均)
        # graph_embed = torch.mean(node_feats, dim=0, keepdim=True)
        graph_embed = torch.max(node_feats, dim=0, keepdim=True)[0]
        graph_embed = self.readout(graph_embed)
        # 通过readout网络处理分子级图特征
        node_readout = self.readout(node_feats)  # [num_nodes, output_dim]
        return node_readout, graph_embed

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
        self.h_out = 4
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
        self.protein_mlp1 = nn.Sequential(
            nn.Linear(protein_emb, output_dim),
            nn.ReLU(),
        )
        # Final prediction MLP
        self.final_mlp = nn.Sequential(
            nn.Linear(self.output_dim*2, output_dim),
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
        # 轻量化注意力机制
        self.feature_conv = nn.Conv1d(self.protein_dim,self.protein_dim,self.p_kernel_size,
                                      stride=self.stride,padding=self.p_kernel_size // 2)
        self.attention_convolution = nn.Conv1d(self.protein_dim,self.protein_dim,self.p_kernel_size,
                                      stride=self.stride,padding=self.p_kernel_size // 2)
        # 双线性注意力机制
        self.bcn = weight_norm(
            BANLayer(v_dim=self.output_dim, q_dim=self.output_dim, h_dim=512, h_out=self.h_out),
            name='h_mat', dim=None)
        self.bcn1 = weight_norm(
            BANLayer(v_dim=self.output_dim, q_dim=self.output_dim, h_dim=512, h_out=self.h_out),
            name='h_mat', dim=None)

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
        graph_drug_embs = []
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
                drug_emb,graph_drug_emb = self.gnn(node_feats, edge_index, edge_feats)
            drug_embs.append(drug_emb.unsqueeze(0))
            graph_drug_embs.append(graph_drug_emb)

        # 堆叠所有药物嵌入
        drug_emb = torch.cat(drug_embs, dim=0)  # (batch_size, num_node, drug_dim)
        graph_drug_emb = torch.cat(graph_drug_embs, dim=0) # (batch_size, drug_dim)
        # print("0", drug_emb.shape)
        # 2. 处理概率向量和蛋白质表征
        prob_weights = self.probability_mlp(probability_vec)  # (batch_size, probability_dim)
        # print("1", prob_weights.shape)
        prob_weights = prob_weights.unsqueeze(-1)   # (batch_size, probability_dim, 1)
        # print("2", prob_weights.shape)
        prob_weights = prob_weights.unsqueeze(-1)   # (batch_size, probability_dim, 1, 1)
        prob_weights = prob_weights.permute(0, 3, 2, 1)   # (batch_size, 1, 1, probability_dim)
        prob_weights_expanded = prob_weights.expand(-1, 4, 46, -1)  # (batch_size, probability_dim, 256)
        # print("3", protein_emb.shape)
        protein_emb_lowdim = self.protein_mlp(protein_emb)  # (batch_size, probability_dim, 256)
        # print("4", prob_weights_expanded.shape)
        # print("5", protein_emb.shape)
        # 3. 加权蛋白质表征
        # weighted_protein = protein_emb_lowdim * prob_weights_expanded  # (batch_size, seq_len, protein_emb)
        # print("6", weighted_protein.shape)
        # 4、双线性注意力机制
        f1, att = self.bcn(v=drug_emb, q=protein_emb_lowdim, p=prob_weights_expanded)
        # print(f1.shape)
        # 8. 最终预测
        affinity = self.final_mlp(f1)  # (batch_size, 1)
        return affinity.squeeze(-1)