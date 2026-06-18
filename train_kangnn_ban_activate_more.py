import random
import utils1.utils
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
import pickle
import argparse
from sklearn.metrics import r2_score
import utils1.utils
from model3_ban_kangnn import LBSDTIAModel
from torch import nn
from utils1.utils import *
import dgl
import copy
import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr


class DTADataset(Dataset):
    def __init__(self, data_list, drug_smiles_dict, drug_embedding_dict):
        self.data = data_list
        self.drug_smiles_dict = drug_smiles_dict
        self.drug_embedding_dict = drug_embedding_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cid, protein_feat, prob_feat, label = self.data[idx]
        smiles = self.drug_smiles_dict.get(str(cid), "")

        if smiles in self.drug_embedding_dict:
            drug_data = self.drug_embedding_dict[smiles]
            dgl_graph = drug_data['graph']
        else:
            if str(cid) in self.drug_embedding_dict:
                drug_data = self.drug_embedding_dict[str(cid)]
                dgl_graph = drug_data['graph']
            else:
                raise KeyError(f"No DGL graph found for CID: {cid}, SMILES: {smiles}")

        return {
            'protein_feat': torch.FloatTensor(protein_feat),
            'prob_feat': torch.FloatTensor(prob_feat),
            'dgl_graph': dgl_graph,
            'label': torch.FloatTensor([label]),
            'cid': cid,
            'smiles': smiles
        }


# --------------------------------------------------------------------------
# 1. 修改后的辅助函数：支持 uncertainty, random, greedy 三种策略
# --------------------------------------------------------------------------
def query_samples(model, unlabeled_data, drug_smiles_dict, drug_embedding,
                  batch_size, device, strategy='uncertainty', n_samples_mc=10, query_size=100):
    """
    Args:
        strategy: 'uncertainty' (默认), 'random', 'greedy'
    """
    # === 策略 1: 随机采样 (Random) ===
    # 不需要模型预测，直接从索引中随机抽
    if strategy == 'random':
        # print("Strategy: Random Sampling") # 保持输出简洁，这里不打印额外内容
        all_indices = list(range(len(unlabeled_data)))
        selected_indices = random.sample(all_indices, min(query_size, len(all_indices)))
        return selected_indices, None

    # === 准备数据加载器 (Greedy 和 Uncertainty 都需要模型推理) ===
    model.eval()
    unlabeled_dataset = DTADataset(unlabeled_data, drug_smiles_dict, drug_embedding)

    def collate_fn(batch):
        protein_feats = torch.stack([item['protein_feat'] for item in batch])
        prob_feats = torch.stack([item['prob_feat'] for item in batch])
        dgl_graphs = [item['dgl_graph'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        return {'protein_feat': protein_feats, 'prob_feat': prob_feats, 'dgl_graph': dgl_graphs, 'label': labels}

    loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, drop_last=False)

    scores = []

    # === 策略 2: 贪婪采样 (Greedy) - 挑选预测值最高的 ===
    if strategy == 'greedy':
        # print("Strategy: Greedy (High Prediction) Sampling")
        with torch.no_grad():
            for batch in loader:
                protein_feat = batch['protein_feat'].to(device)
                prob_feat = batch['prob_feat'].to(device)
                dgl_graphs = batch['dgl_graph']
                batched_graph = dgl.batch(dgl_graphs).to(device)

                # 直接预测，无需 MC Dropout
                preds = model(protein_feat, batched_graph, prob_feat)
                # 展平并存入
                scores.extend(preds.cpu().numpy().flatten())

        scores = np.array(scores)
        # argsort 从小到大排，[::-1] 反转取最大的（高亲和力）
        sorted_indices = np.argsort(scores)[::-1]
        selected_indices = sorted_indices[:query_size]

        return selected_indices, scores

    # === 策略 3: 不确定性采样 (Uncertainty) - 原始逻辑 ===
    elif strategy == 'uncertainty':
        # print("Strategy: Uncertainty (Variance) Sampling")
        print(f"Scanning unlabeled pool ({len(unlabeled_data)} samples) for uncertainty...")

        with torch.no_grad():
            for batch in loader:
                protein_feat = batch['protein_feat'].to(device)
                prob_feat = batch['prob_feat'].to(device)
                dgl_graphs = batch['dgl_graph']

                batch_preds = []
                for _ in range(n_samples_mc):
                    batched_graph = dgl.batch(dgl_graphs).to(device)
                    # 开启 return_dual 获取带噪声的预测
                    _, noisy_pred = model(protein_feat, batched_graph, prob_feat, return_dual=True)
                    batch_preds.append(noisy_pred)
                    del batched_graph
                    torch.cuda.empty_cache()

                batch_preds = torch.stack(batch_preds)
                # 计算方差
                uncertainty = batch_preds.var(dim=0).cpu().numpy().flatten()
                scores.extend(uncertainty)

        scores = np.array(scores)
        # 选方差最大的
        sorted_indices = np.argsort(scores)[::-1]
        selected_indices = sorted_indices[:query_size]

        return selected_indices, scores

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_model(predictions, targets):
    mse = utils1.utils.mse(targets, predictions)
    rmse = utils1.utils.rmse(targets, predictions)
    spearman = utils1.utils.spearman(targets, predictions)
    pearson = utils1.utils.pearson(targets, predictions)
    r2 = r2_score(targets, predictions)
    ci = utils1.utils.ci(targets, predictions)
    return {'mse': mse, 'rmse': rmse, 'r2': r2, 'spearman': spearman, 'pearson': pearson, 'ci': ci}


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    start_time = time.time()

    with torch.no_grad():
        for batch in dataloader:
            protein_feat = batch['protein_feat'].to(device)
            prob_feat = batch['prob_feat'].to(device)
            dgl_graphs = batch['dgl_graph']
            labels = batch['label'].to(device).squeeze()
            batched_graph = dgl.batch(dgl_graphs).to(device)

            outputs = model(protein_feat, batched_graph, prob_feat)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    epoch_time = time.time() - start_time
    return total_loss / len(dataloader), np.array(predictions), np.array(targets), epoch_time


def validate_with_uncertainty(model, dataloader, device, n_samples=10):
    model.eval()
    all_mean_preds = []
    all_uncertainties = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            protein_feat = batch['protein_feat'].to(device)
            prob_feat = batch['prob_feat'].to(device)
            dgl_graphs = batch['dgl_graph']
            labels = batch['label'].to(device).squeeze()

            batch_preds = []
            for _ in range(n_samples):
                batched_graph = dgl.batch(dgl_graphs).to(device)
                _, noisy_pred = model(protein_feat, batched_graph, prob_feat, return_dual=True)
                batch_preds.append(noisy_pred)
                del batched_graph
                torch.cuda.empty_cache()

            batch_preds = torch.stack(batch_preds)
            mean_pred = batch_preds.mean(dim=0)
            uncertainty = batch_preds.var(dim=0)

            all_mean_preds.extend(mean_pred.cpu().numpy())
            all_uncertainties.extend(uncertainty.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    return np.array(all_mean_preds), np.array(all_targets), np.array(all_uncertainties)


def calc_uncertainty_correlation(predictions, targets, uncertainties):
    abs_errors = np.abs(predictions - targets)
    corr, p_value = pearsonr(abs_errors, uncertainties)
    return corr, abs_errors.mean()


# --------------------------------------------------------------------------
# 2. 主动学习训练主流程
# --------------------------------------------------------------------------
def active_learning_train(args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Strategy: {args.strategy}")  # 打印当前运行的策略

    # ================= 1. 数据准备 =================
    lbs_path = "./dataset/dtadelete/" + args.dataset + "/lbsprediction.tsv"
    fold_idx = 3
    folds_path = "./dataset/dtadelete/" + args.dataset + "/data_folds/" + args.setting + "/"
    data_path = "./dataset/dtadelete/" + args.dataset + "/"
    drug_embedding_path = data_path + "drug_emb/" + "drug_graphs.pth"

    drug_smiles_dict = pickle.load(open(data_path + "drug_smiles_kiba.pkl", "rb"))
    drug_embedding = torch.load(drug_embedding_path)
    prot_feat = pickle.load(open(data_path + "protein_emb/protein_features_prostt5.pkl", "rb"))

    full_train_data, test_data = utils1.utils.load_data(folds_path, fold_idx, lbs_path, prot_feat)

    target_ratios = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
    n_cycles = len(target_ratios) - 1

    total_train_size = len(full_train_data)
    random.shuffle(full_train_data)

    initial_count = int(total_train_size * target_ratios[0])
    labeled_pool = full_train_data[:initial_count]
    unlabeled_data = full_train_data[initial_count:]

    print(f"Initial Split: Labeled={len(labeled_pool)}, Unlabeled={len(unlabeled_data)}")

    test_dataset = DTADataset(test_data, drug_smiles_dict, drug_embedding)

    def collate_fn(batch):
        protein_feats = torch.stack([item['protein_feat'] for item in batch])
        prob_feats = torch.stack([item['prob_feat'] for item in batch])
        dgl_graphs = [item['dgl_graph'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        return {'protein_feat': protein_feats, 'prob_feat': prob_feats, 'dgl_graph': dgl_graphs, 'label': labels}

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    al_performance_history = []

    # ================= 2. 主动学习循环 (AL Loop) =================
    for cycle in range(n_cycles + 1):
        print(f"\n{'=' * 20} AL Cycle {cycle} (Target: {target_ratios[cycle] * 100}%) {'=' * 20}")
        print(f"Current Labeled Size: {len(labeled_pool)} / Total: {total_train_size}")

        # --- A. 模型初始化 ---
        protein_feat_dim = labeled_pool[0][1].shape[1]
        model = LBSDTIAModel(
            protein_emb=protein_feat_dim, probability_dim=1500,
            gnn_hidden_dim=128, gnn_output_dim=256, hidden_dim=128, output_dim=256, max_nodenum=83
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()

        # --- B. 训练 ---
        train_dataset = DTADataset(labeled_pool, drug_smiles_dict, drug_embedding)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        best_test_mse = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())

        patience = 70
        counter = 0
        eval_gap = 5
        max_epochs = 700

        model.train()
        for epoch in range(max_epochs):
            total_train_loss = 0
            for batch in train_loader:
                protein_feat = batch['protein_feat'].to(device)
                prob_feat = batch['prob_feat'].to(device)
                batched_graph = dgl.batch(batch['dgl_graph']).to(device)
                labels = batch['label'].to(device).squeeze()

                optimizer.zero_grad()
                outputs = model(protein_feat, batched_graph, prob_feat)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            if epoch % eval_gap == 0:
                current_test_mse, _, _, _ = validate_epoch(model, test_loader, criterion, device)

                if current_test_mse < best_test_mse:
                    best_test_mse = current_test_mse
                    best_model_wts = copy.deepcopy(model.state_dict())
                    counter = 0
                else:
                    counter += 1

                if counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        # --- C. 评估 (严格仿照给定的打印格式) ---
        print("Evaluating on Test Set with Uncertainty Estimation...")
        model.load_state_dict(best_model_wts)
        print("Finalizing Cycle with Best Model Weights...")

        test_pred, test_true, test_uncertainty = validate_with_uncertainty(
            model, test_loader, device, n_samples=20
        )

        metrics = evaluate_model(test_pred, test_true)
        unc_corr, test_mae = calc_uncertainty_correlation(test_pred, test_true, test_uncertainty)

        # ----------------------------------------------------
        # 严格保持你要求的输出格式
        # ----------------------------------------------------
        print(f"Cycle {cycle} Results:")
        print(f"  MSE: {metrics['mse']:.4f}, R2: {metrics['r2']:.4f}, CI: {metrics['ci']:.4f}")
        print(f"  Pearson: {metrics['pearson']:.4f}, spearman: {metrics['spearman']:.4f}, rmse: {metrics['rmse']:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  Uncertainty Correlation (Pearson): {unc_corr:.4f}")
        # ----------------------------------------------------

        al_performance_history.append({
            'cycle': cycle,
            'labeled_size': len(labeled_pool),
            'mse': metrics['mse'],
            'r2': metrics['r2'],
            'ci': metrics['ci'],
            'mae': test_mae,
            'unc_correlation': unc_corr
        })

        # --- D. Query (采样) ---
        if cycle < n_cycles:
            next_target_count = int(total_train_size * target_ratios[cycle + 1])
            current_query_size = next_target_count - len(labeled_pool)

            if current_query_size <= 0 or len(unlabeled_data) == 0:
                print("Reach maximum capacity or no more unlabeled data.")
                break

            print(f"Querying {current_query_size} new samples to reach {target_ratios[cycle + 1] * 100}%...")

            # --- 关键修改：传入 args.strategy ---
            selected_indices, _ = query_samples(
                model=model,
                unlabeled_data=unlabeled_data,
                drug_smiles_dict=drug_smiles_dict,
                drug_embedding=drug_embedding,
                batch_size=args.batch_size,
                device=device,
                strategy=args.strategy,  # 使用命令行参数指定的策略
                n_samples_mc=10,
                query_size=current_query_size
            )

            new_labeled = [unlabeled_data[i] for i in selected_indices]
            labeled_pool.extend(new_labeled)

            selected_indices_set = set(selected_indices)
            unlabeled_data = [d for i, d in enumerate(unlabeled_data) if i not in selected_indices_set]

            print(f"Selected {len(new_labeled)} samples.")

    # ================= 3. 保存结果 =================
    df_history = pd.DataFrame(al_performance_history)
    # 保存时文件名带上策略名称，防止覆盖
    save_path = os.path.join(args.output_dir, f"{args.dataset}_{args.strategy}_AL_history.csv")
    df_history.to_csv(save_path, index=False)
    print(f"Active Learning History saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="davis")
    parser.add_argument('--setting', type=str, default="warm_start")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output_dir', type=str, default='./result_al')
    # 新增参数：策略选择
    # uncertainty: 不确定性采样 (默认)
    # random: 随机采样
    # greedy: 贪婪采样 (高预测值优先)
    parser.add_argument('--strategy', type=str, default='uncertainty',
                        choices=['uncertainty', 'random', 'greedy'],
                        help='Active learning query strategy')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    active_learning_train(args)