import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()
        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        # --- 修改建议：使用 Xavier 初始化替代 Normal 以防止梯度爆炸/消失 ---
        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k))
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1))
            nn.init.xavier_normal_(self.h_mat)  # 推荐初始化
            nn.init.constant_(self.h_bias, 0)
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k
        return fusion_logits

    def forward(self, v, q, softmax=True, v_mask=None, q_mask=None):
        """
        Args:
            v: [batch, v_num, dim] (药物原子)
            q: [batch, q_num, dim] (蛋白质序列)
            v_mask: [batch, v_num] 0/1 掩码
            q_mask: [batch, q_num] 0/1 掩码 (新增)
        """
        v_num = v.size(1)
        q_num = q.size(1)

        # 1. 特征变换
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)

        # 2. 【关键修改】构建双向联合 Mask
        # 我们需要屏蔽掉 (Pad_Drug, Real_Prot), (Real_Drug, Pad_Prot), (Pad_Drug, Pad_Prot)
        if v_mask is not None or q_mask is not None:
            # 如果某一方没有提供 mask，默认为全 1 (即全有效)
            if v_mask is None:
                v_mask = torch.ones(v.size(0), v_num, device=v.device)
            if q_mask is None:
                q_mask = torch.ones(v.size(0), q_num, device=q.device)

            # 计算外积: [Batch, V] * [Batch, Q] -> [Batch, V, Q]
            # 结果为 1 表示两者都是真实数据，结果为 0 表示至少有一方是 Padding
            joint_mask = torch.einsum('bi,bj->bij', v_mask, q_mask)

            # 扩展维度以匹配 att_maps: [Batch, 1, V, Q]
            joint_mask = joint_mask.unsqueeze(1)

            # 将无效位置的 Attention Score 设为极小负数
            att_maps = att_maps.masked_fill(joint_mask == 0, -1e9)

        # 3. Softmax
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)

        # 4. Pooling
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i

        logits = self.bn(logits)
        return logits, att_maps


class FCNet(nn.Module):
    # ... (保持原本 FCNet 代码不变) ...
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()
        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)