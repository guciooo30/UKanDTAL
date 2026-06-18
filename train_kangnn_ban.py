import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import time
import pickle
import argparse
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import utils1.utils
from model3_ban_kangnn import LBSDTIAModel
from torch import nn
from utils1.utils import *
import dgl


# set dataset class
class DTADataset(Dataset):
    def __init__(self, data_list, drug_smiles_dict, drug_embedding_dict):
        self.data = data_list
        self.drug_smiles_dict = drug_smiles_dict
        self.drug_embedding_dict = drug_embedding_dict  # 存储药物DGL图特征的字典

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cid, protein_feat, prob_feat, label = self.data[idx]
        smiles = self.drug_smiles_dict.get(str(cid), "")

        # 根据SMILES从字典中获取对应的DGL图
        if smiles in self.drug_embedding_dict:
            drug_data = self.drug_embedding_dict[smiles]
            dgl_graph = drug_data['graph']  # 提取DGL图对象
        else:
            # 如果没有找到对应的图，尝试直接使用CID查找
            if str(cid) in self.drug_embedding_dict:
                drug_data = self.drug_embedding_dict[str(cid)]
                dgl_graph = drug_data['graph']
            else:
                # 如果仍然找不到，抛出一个更详细的错误
                raise KeyError(f"No DGL graph found for CID: {cid}, SMILES: {smiles}")

        return {
            'protein_feat': torch.FloatTensor(protein_feat),
            'prob_feat': torch.FloatTensor(prob_feat),
            'dgl_graph': dgl_graph,  # 返回DGL图对象
            'label': torch.FloatTensor([label]),
            'cid': cid,  # 保留CID用于调试
            'smiles': smiles  # 保留SMILES用于调试
        }


# train
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    beta1 = 0.01
    predictions = []
    targets = []
    start_time = time.time()

    for batch in dataloader:
        protein_feat = batch['protein_feat'].to(device)
        prob_feat = batch['prob_feat'].to(device)
        dgl_graphs = batch['dgl_graph']  # 获取DGL图列表
        labels = batch['label'].to(device).squeeze()

        optimizer.zero_grad()

        # 将DGL图批处理并转移到设备
        batched_graph = dgl.batch(dgl_graphs).to(device)

        # 将批处理的DGL图传递给模型
        outputs = model(protein_feat, batched_graph, prob_feat)
        loss = criterion(outputs, labels)
        # outputs, att_global, att_local = model(protein_feat, batched_graph, prob_feat)
        # loss1 = criterion(outputs, labels)
        # loss2 = gcl_loss(att_global, att_local)
        # loss = loss1 + loss2 * beta1
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions.extend(outputs.detach().cpu().numpy())
        targets.extend(labels.detach().cpu().numpy())

    epoch_time = time.time() - start_time
    return total_loss / len(dataloader), np.array(predictions), np.array(targets), epoch_time


# val
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    beta1 = 0.01
    predictions = []
    targets = []
    start_time = time.time()

    with torch.no_grad():
        for batch in dataloader:
            protein_feat = batch['protein_feat'].to(device)
            prob_feat = batch['prob_feat'].to(device)
            dgl_graphs = batch['dgl_graph']  # 获取DGL图列表
            labels = batch['label'].to(device).squeeze()

            # 将DGL图批处理并转移到设备
            batched_graph = dgl.batch(dgl_graphs).to(device)

            # 将批处理的DGL图传递给模型
            outputs = model(protein_feat, batched_graph, prob_feat)
            loss = criterion(outputs, labels)
            # outputs, att_global, att_local = model(protein_feat, batched_graph, prob_feat)
            # loss1 = criterion(outputs, labels)
            # loss2 = gcl_loss(att_global, att_local)
            # loss = loss1 + loss2 * beta1

            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy().reshape(-1))

    epoch_time = time.time() - start_time
    return total_loss / len(dataloader), np.array(predictions), np.array(targets), epoch_time


# save metrics into csv
def save_metrics(metrics, filename):
    """保存评估指标到CSV文件"""
    df = pd.DataFrame([metrics])
    df.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")


def create_directories(output_dir, i):
    """创建必要的目录结构"""
    metric_dir = os.path.join(output_dir, "metric" + str(i))
    model_dir = os.path.join(output_dir, "models" + str(i))

    os.makedirs(metric_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    return metric_dir, model_dir


# 绘制并保存损失曲线
def save_loss_plot(train_losses, val_losses, filename):
    """绘制并保存损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {filename}")


# 保存训练历史
def save_training_history(train_losses, val_losses, train_metrics_history, val_metrics_history, train_times, val_times
                          , filename):
    """保存完整的训练历史到CSV文件"""
    history_data = []
    for epoch, (train_loss, val_loss, train_metrics, val_metrics, train_time, val_time) in enumerate(
            zip(train_losses, val_losses, train_metrics_history, val_metrics_history, train_times, val_times)
    ):
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_mse': train_metrics['mse'],
            'train_rmse': train_metrics['rmse'],
            'train_r2': train_metrics['r2'],
            'train_spearman': train_metrics['spearman'],
            'train_pearson': train_metrics['pearson'],
            'val_mse': val_metrics['mse'],
            'val_rmse': val_metrics['rmse'],
            'val_r2': val_metrics['r2'],
            'val_spearman': val_metrics['spearman'],
            'val_pearson': val_metrics['pearson'],
            'train_ci': train_metrics['ci'],
            'val_ci': val_metrics['ci'],
            'train_time_s': train_time,
            'val_time_s': val_time,
        }
        history_data.append(epoch_data)

    df = pd.DataFrame(history_data)
    df.to_csv(filename, index=False)
    print(f"Training history saved to {filename}")


# 评估函数
def evaluate_model(predictions, targets):
    mse = utils1.utils.mse(targets, predictions)
    rmse = utils1.utils.rmse(targets, predictions)
    spearman = utils1.utils.spearman(targets, predictions)
    pearson = utils1.utils.pearson(targets, predictions)
    r2 = r2_score(targets, predictions)
    ci = utils1.utils.ci(targets, predictions)
    return {'mse': mse, 'rmse': rmse, 'r2': r2, 'spearman': spearman, 'pearson': pearson, 'ci': ci}


def start_train(k_folds: int, args) -> None:
    for i in range(1, 5):
        print("fold:", i + 1, "/", k_folds)
        # 设置设备
        device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        if i > 1:
            break

        # 数据路径
        lbs_path = "./dataset/dtadelete/" + args.dataset + "/lbsprediction.tsv"
        folds_path = "./dataset/dtadelete/" + args.dataset + "/data_folds/" + args.setting + "/"
        data_path = "./dataset/dtadelete/" + args.dataset + "/"
        drug_embedding_path = data_path + "drug_emb/" + "drug_graphs.pth"
        # 加载药物SMILES字典
        drug_smiles_dict = pickle.load(open(data_path + "drug_smiles_kiba.pkl", "rb"))
        # 加载药物的dgl图表征
        drug_embedding = torch.load(drug_embedding_path)
        # 加载蛋白质特征
        prot_feat = pickle.load(open(data_path + "protein_emb/protein_features_prostt5.pkl", "rb"))
        # 加载数据
        train_data, test_data = utils1.utils.load_data(folds_path, i, lbs_path, prot_feat)
        # 设置保存路径
        output_dir = args.output_dir + "/" + args.dataset + "/" + args.setting
        metric_dir, model_dir = create_directories(output_dir, i)
        # 创建数据集和数据加载器
        train_dataset = DTADataset(train_data, drug_smiles_dict, drug_embedding)
        test_dataset = DTADataset(test_data, drug_smiles_dict, drug_embedding)

        # 自定义collate_fn函数
        def collate_fn(batch):
            protein_feats = torch.stack([item['protein_feat'] for item in batch])
            prob_feats = torch.stack([item['prob_feat'] for item in batch])
            dgl_graphs = [item['dgl_graph'] for item in batch]
            labels = torch.stack([item['label'] for item in batch])

            return {
                'protein_feat': protein_feats,
                'prob_feat': prob_feats,
                'dgl_graph': dgl_graphs,
                'label': labels
            }

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                                  pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                 pin_memory=False)

        # 初始化模型
        protein_feat_dim = train_data[0][1].shape[1]  # 获取蛋白质特征维度
        probability_dim = 1500  # 根据您的数据设置

        """Model for predicting protein-drug binding affinity with GNN for drug representation"""
        """
        davis_data's max_numnode is 83,kiba_data's max_numnode is 194
        """
        model = LBSDTIAModel(
            protein_emb=protein_feat_dim,
            probability_dim=probability_dim,
            gnn_hidden_dim=128,
            gnn_output_dim=256,
            hidden_dim=128,
            output_dim=256,
            max_nodenum=194
        ).to(device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # 监控指标下降时调整
            patience=15,  # 10个epoch指标不下降则减小LR
            factor=0.5,  # 学习率减半
        )
        start_epoch = 0
        best_val_loss = float('inf')
        if args.resume_training and args.checkpoint_path:
            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['val_loss']
            print(f"Resumed training from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")

        # 训练循环
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        train_metrics_history = []
        val_metrics_history = []
        train_times = []
        val_times = []
        early_stop_counter = 0
        patience = 50
        print("Starting training...")
        for epoch in range(start_epoch, args.epochs):
            train_loss, train_pred, train_true, train_time = train_epoch(model, train_loader, optimizer, criterion,
                                                                         device)
            val_loss, val_pred, val_true, val_time = validate_epoch(model, test_loader, criterion, device)

            train_times.append(train_time)
            val_times.append(val_time)

            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # 计算评估指标
            # train_metrics = evaluate_model(train_pred, train_true)
            # val_metrics = evaluate_model(val_pred, val_true)
            # train_metrics_history.append(train_metrics)
            # val_metrics_history.append(val_metrics)

            print(f'Epoch {epoch + 1}/{args.epochs}:')
            print(f'  Train Time: {train_time:.2f}s, Val Time: {val_time:.2f}s')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            # print(f'    MSE: {val_metrics["mse"]:.4f}, R²: {val_metrics["r2"]:.4f}, '
            #       f'Spearman: {val_metrics["spearman"]:.4f}, Pearson: {val_metrics["pearson"]:.4f}')
            # print(f'    RMSE: {val_metrics["rmse"]:.4f}, CI: {val_metrics["ci"]:.4f}')

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, os.path.join(model_dir, f'best_model_fold_{i}.pth'))
                # save_metrics(train_metrics, os.path.join(metric_dir, f'best_final_train_metrics_fold_{i}.csv'))
                val_metrics = evaluate_model(val_pred, val_true)
                val_metrics_history.append(val_metrics)
                print(f'    MSE: {val_metrics["mse"]:.4f}, R²: {val_metrics["r2"]:.4f}, '
                      f'Spearman: {val_metrics["spearman"]:.4f}, Pearson: {val_metrics["pearson"]:.4f}')
                print(f'    RMSE: {val_metrics["rmse"]:.4f}, CI: {val_metrics["ci"]:.4f}')
                save_metrics(val_metrics, os.path.join(metric_dir, f'best_final_val_metrics_fold_{i}.csv'))
                early_stop_counter = 0
                print(f'  Saved best model with val_loss: {val_loss:.4f}')
            else:
                early_stop_counter += 1

            # 检查早停条件
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1} (fold {i + 1})")
                break

        # 绘制并保存损失曲线
        save_loss_plot(train_losses, val_losses, os.path.join(metric_dir, f'training_loss_fold_{i}.png'))

        # 保存完整的训练历史
        save_training_history(
            train_losses, val_losses, train_metrics_history, val_metrics_history, train_times, val_times,
            os.path.join(metric_dir, f'training_history_fold_{i}.csv')
        )

        print(f"Training completed for fold {i + 1}!")


# 主训练函数
def main():
    parser = argparse.ArgumentParser(description='LBS-DTI Model Training')
    parser.add_argument('--setting', type=str, default="warm_start", help='Task setting Type')
    parser.add_argument('--dataset', type=str, default="davis", help='Dataset name')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.004, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./result', help='Path to output directory')
    parser.add_argument('--checkpoint_path', type=str, default='', help='Path to checkpoint for resuming training')
    parser.add_argument('--resume_training', action='store_true', default=False, help='Resume training')
    args = parser.parse_args()
    k_folds = 5
    start_train(k_folds, args)


if __name__ == "__main__":
    main()
