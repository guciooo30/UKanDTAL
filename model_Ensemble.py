import dgl
import torch
import numpy as np
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import GINConv, global_add_pool

RDLogger.DisableLog('rdApp.*')


# ========== 原有的原子特征和SMILES处理函数 ==========
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
    """DeepDTA版本的原子特征"""
    try:
        implicit_valence = atom.GetImplicitValence()
    except Exception as e:
        print(f"Warning: GetImplicitValence failed: {e}, using default value 0")
        implicit_valence = 0

    features = np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
            'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
            'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
            'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
        ]) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(implicit_valence, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()]
    )
    return features


def smiles_to_graph(smiles, device='cpu'):
    """使用DeepDTA的原子特征，支持设备指定"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None, None

    atom_features_list = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        feature_sum = sum(feature)
        if feature_sum > 0:
            normalized_feature = feature / feature_sum
        else:
            normalized_feature = feature
        atom_features_list.append(normalized_feature)

    atom_features_array = np.array(atom_features_list)
    node_feats = torch.tensor(atom_features_array, dtype=torch.float32).to(device)

    src, dst = [], []
    edge_features = []
    for bond in mol.GetBonds():
        src.append(bond.GetBeginAtomIdx())
        dst.append(bond.GetEndAtomIdx())
        bond_features = [
            float(bond.GetBondTypeAsDouble()),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            float(bond.GetStereo()),
        ]
        edge_features.append(bond_features)

    if src and dst:
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long).to(device)
    else:
        edge_index = torch.tensor([[], []], dtype=torch.long).to(device)

    if edge_features:
        edge_features_array = np.array(edge_features * 2)
        edge_feats = torch.tensor(edge_features_array, dtype=torch.float32).to(device)
    else:
        edge_feats = None

    return node_feats, edge_index, edge_feats


# ========== 集成学习模块 ==========
class EnsembleLayer(nn.Module):
    """集成学习层，融合多个基学习器的预测"""

    def __init__(self, input_dim, ensemble_method='stacking', n_models=5, use_metalearning=True):
        super(EnsembleLayer, self).__init__()
        self.ensemble_method = ensemble_method
        self.n_models = n_models
        self.use_metalearning = use_metalearning
        self.input_dim = input_dim

        # 初始化基学习器
        if ensemble_method == 'stacking':
            self.base_models = nn.ModuleList([
                self._create_base_model(input_dim) for _ in range(n_models)
            ])
            # 元学习器
            self.meta_learner = nn.Sequential(
                nn.Linear(n_models, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )

        elif ensemble_method == 'voting':
            self.base_models = nn.ModuleList([
                self._create_base_model(input_dim) for _ in range(n_models)
            ])
            # 加权投票的权重
            self.voting_weights = nn.Parameter(torch.ones(n_models) / n_models)

        elif ensemble_method == 'bagging':
            self.base_models = nn.ModuleList([
                self._create_base_model(input_dim) for _ in range(n_models)
            ])

        elif ensemble_method == 'gradient_boosting':
            # 梯度提升的弱学习器
            self.weak_learners = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                ) for _ in range(n_models)
            ])
            self.learning_rate = 0.1

        elif ensemble_method == 'mixture_of_experts':
            # 专家混合模型
            self.experts = nn.ModuleList([
                self._create_base_model(input_dim) for _ in range(n_models)
            ])
            # 门控网络
            self.gate_network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n_models),
                nn.Softmax(dim=-1)
            )
        else:
            # 默认：简单平均
            self.base_models = nn.ModuleList([
                self._create_base_model(input_dim) for _ in range(n_models)
            ])

    def _create_base_model(self, input_dim):
        """创建基学习器"""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, training=True):
        """前向传播"""
        batch_size = x.shape[0]

        if self.ensemble_method == 'stacking':
            # 1. 第一阶段：基学习器预测
            base_predictions = []
            for model in self.base_models:
                pred = model(x)
                base_predictions.append(pred)

            # 堆叠预测结果
            stacked = torch.stack(base_predictions, dim=1).squeeze(-1)  # [batch_size, n_models]

            # 2. 第二阶段：元学习器
            if training and self.use_metalearning:
                # 训练时使用dropout增强多样性
                stacked = F.dropout(stacked, p=0.2, training=True)

            output = self.meta_learner(stacked)

        elif self.ensemble_method == 'voting':
            # 加权投票
            predictions = []
            weights = F.softmax(self.voting_weights, dim=0)

            for i, model in enumerate(self.base_models):
                pred = model(x) * weights[i]
                predictions.append(pred)

            output = torch.sum(torch.stack(predictions), dim=0)

        elif self.ensemble_method == 'bagging':
            # Bootstrap聚合
            predictions = []
            for model in self.base_models:
                if training:
                    # 训练时应用不同的dropout mask增加多样性
                    x_aug = F.dropout(x, p=0.2, training=True)
                    pred = model(x_aug)
                else:
                    pred = model(x)
                predictions.append(pred)

            output = torch.mean(torch.stack(predictions), dim=0)

        elif self.ensemble_method == 'gradient_boosting':
            # 梯度提升
            output = torch.zeros(batch_size, 1, device=x.device)
            residual = torch.randn_like(output) * 0.1  # 初始残差

            for i, learner in enumerate(self.weak_learners):
                # 学习残差
                pred = learner(x)
                output = output + self.learning_rate * pred

                if training and i < len(self.weak_learners) - 1:
                    # 更新残差
                    residual = residual - self.learning_rate * pred

        elif self.ensemble_method == 'mixture_of_experts':
            # 专家混合
            expert_outputs = []
            for expert in self.experts:
                expert_outputs.append(expert(x))

            expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, n_models, 1]

            # 门控权重
            gate_weights = self.gate_network(x)  # [batch_size, n_models]

            # 加权求和
            output = torch.sum(expert_outputs * gate_weights.unsqueeze(-1), dim=1)

        else:
            # 默认：简单平均
            predictions = [model(x) for model in self.base_models]
            output = torch.mean(torch.stack(predictions), dim=0)

        return output.squeeze(-1)


class DiversityRegularizer(nn.Module):
    """多样性正则化器，增加集成模型的多样性"""

    def __init__(self, diversity_weight=0.1):
        super(DiversityRegularizer, self).__init__()
        self.diversity_weight = diversity_weight

    def forward(self, predictions):
        """
        predictions: List of tensors, each shape [batch_size, 1]
        """
        n_models = len(predictions)
        if n_models < 2:
            return 0.0

        # 计算两两之间的负相关性
        diversity_loss = 0.0
        count = 0

        for i in range(n_models):
            for j in range(i + 1, n_models):
                # 计算相关性
                pred_i = predictions[i] - predictions[i].mean()
                pred_j = predictions[j] - predictions[j].mean()

                correlation = (pred_i * pred_j).sum() / (
                        torch.sqrt((pred_i ** 2).sum()) * torch.sqrt((pred_j ** 2).sum()) + 1e-8
                )

                # 鼓励负相关性
                diversity_loss += torch.relu(correlation + 0.5)  # 惩罚高正相关
                count += 1

        return self.diversity_weight * (diversity_loss / count) if count > 0 else 0.0


# ========== 原有的GNN相关类 ==========
class GraphConv(nn.Module):
    """简单的图卷积层实现"""

    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, edge_index):
        row, col = edge_index
        neighbor_agg = scatter_mean(x[row], col, dim=0, dim_size=x.size(0))
        x = self.linear(x + neighbor_agg)
        return F.relu(x)


class GCNLayer(nn.Module):
    """标准的图卷积网络层"""

    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = scatter_add(torch.ones_like(row), row, dim=0, dim_size=x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        messages = x[col] * norm.view(-1, 1)
        aggregated = scatter_add(messages, row, dim=0, dim_size=x.size(0))
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

    def forward(self, node_feats, edge_index, edge_feats=None):
        node_feats = self.node_embed(node_feats)
        if edge_feats is not None:
            edge_feats = self.edge_embed(edge_feats)
        node_feats = self.gnn1(node_feats, edge_index)
        node_feats = self.gnn2(node_feats, edge_index)
        node_feats = self.gnn3(node_feats, edge_index)
        graph_embed = torch.max(node_feats, dim=0, keepdim=True)[0]
        node_readout = self.readout(graph_embed)
        return node_readout


# ========== 修改后的主模型（添加集成学习） ==========
class LBSDTIAModelWithEnsemble(nn.Module):
    """添加集成学习策略的DTI预测模型"""

    def __init__(self,
                 protein_emb=1024,
                 gnn_hidden_dim=128,
                 gnn_output_dim=256,
                 probability_dim=1500,
                 hidden_dim=128,
                 output_dim=256,
                 ensemble_method='stacking',
                 n_ensemble_models=5,
                 use_diversity_regularization=True):

        super(LBSDTIAModelWithEnsemble, self).__init__()

        if probability_dim is None:
            raise ValueError("probability_dim must be specified")

        self.output_dim = output_dim
        self.protein_dim = protein_emb
        self.drug_dim = gnn_output_dim
        self.ensemble_method = ensemble_method

        # 原有组件
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=-1)
        self.batch_norm = nn.BatchNorm1d(output_dim)

        # GNN for processing drug
        self.gnn = GNNModel(
            node_input_dim=78,
            edge_input_dim=4,
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

        # 集成学习层
        combined_features_dim = self.output_dim + self.drug_dim
        self.ensemble_layer = EnsembleLayer(
            input_dim=combined_features_dim,
            ensemble_method=ensemble_method,
            n_models=n_ensemble_models
        )

        # 多样性正则化
        self.use_diversity_regularization = use_diversity_regularization
        if use_diversity_regularization:
            self.diversity_regularizer = DiversityRegularizer(diversity_weight=0.1)

        # 可选：保留原有的单模型预测器作为基准
        self.baseline_mlp = nn.Sequential(
            nn.Linear(combined_features_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(output_dim // 2, 1),
        )

        # 特征融合门控（可选）
        self.fusion_gate = nn.Sequential(
            nn.Linear(combined_features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, protein_emb, drug_smiles_list, probability_vec,
                training=True, return_ensemble_details=False):
        """
        Args:
            protein_emb: Tensor of shape (batch_size, seq_len, protein_emb)
            drug_smiles_list: List of SMILES strings of length batch_size
            probability_vec: Tensor of shape (batch_size, 1, probability_dim)
            training: 是否处于训练模式
            return_ensemble_details: 是否返回集成学习的详细信息
        """
        batch_size = len(drug_smiles_list)
        device = protein_emb.device  # 获取输入数据的设备

        # 1. 处理药物SMILES：转换为图并获取GNN嵌入
        drug_embs = []
        for i in range(batch_size):
            # 关键修复：传递设备参数，确保图数据在正确设备上
            node_feats, edge_index, edge_feats = smiles_to_graph(drug_smiles_list[i], device=device)
            if node_feats is None:
                drug_emb = torch.zeros((1, self.drug_dim), device=device)
            else:
                # 注意：现在smiles_to_graph已经返回正确设备上的tensor
                drug_emb = self.gnn(node_feats, edge_index, edge_feats)
            drug_embs.append(drug_emb)

        drug_emb = torch.cat(drug_embs, dim=0)

        # 2. 处理概率向量和蛋白质表征
        prob_weights = self.probability_mlp(probability_vec)
        prob_weights = prob_weights.unsqueeze(-1)
        prob_weights_expanded = prob_weights.expand(-1, -1, self.output_dim)
        protein_emb_lowdim = self.protein_mlp(protein_emb)
        weighted_protein = protein_emb_lowdim * prob_weights_expanded
        prob_protein = weighted_protein.sum(dim=1)

        prob_protein = self.batch_norm(prob_protein)
        drug_emb = self.batch_norm(drug_emb)

        # 3. 拼接蛋白质和药物表征
        combined = torch.cat([prob_protein, drug_emb], dim=1)

        # 4. 使用集成学习进行预测
        ensemble_pred = self.ensemble_layer(combined, training=training)

        # 5. 可选：与基线模型融合
        baseline_pred = self.baseline_mlp(combined).squeeze(-1)

        # 6. 门控融合（可选）
        gate_weights = self.fusion_gate(combined)
        final_pred = gate_weights[:, 0:1] * ensemble_pred + gate_weights[:, 1:2] * baseline_pred

        if return_ensemble_details:
            return {
                'prediction': final_pred,
                'ensemble_pred': ensemble_pred,
                'baseline_pred': baseline_pred,
                'gate_weights': gate_weights,
                'combined_features': combined.detach()
            }

        return final_pred

    def get_diversity_loss(self, combined_features):
        """计算多样性损失（用于训练）"""
        if not self.use_diversity_regularization:
            return 0.0

        # 获取所有基学习器的预测
        predictions = []
        if hasattr(self.ensemble_layer, 'base_models'):
            for model in self.ensemble_layer.base_models:
                pred = model(combined_features)
                predictions.append(pred)

        if len(predictions) > 1:
            return self.diversity_regularizer(predictions)
        return 0.0
