import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import util
from model import model_builder
from data_loader import downstream_dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="Glyph.*missing from font")
import torch
import torch.nn as nn
import torch.nn.functional as F

# 简单的高效 1D-CNN 模块，用于“原生通路”
class RawSignalPath(nn.Module):
    def __init__(self, input_channels, out_dim):
        super().__init__()
        # 使用多尺度卷积或者简单的残差块
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):
        # x: [B, 1, L]
        feat = self.feature_extractor(x).squeeze(-1) # [B, 128]
        return self.proj(feat) # [B, out_dim]

class PPGDualPathModel(nn.Module):
    def __init__(self, pretrained_encoder, d_model, target_points=1000, is_classification=False):
        super().__init__()
        self.encoder = pretrained_encoder # 通路 A: 预训练专家
        self.target_points = int(target_points)
        self.is_classification = is_classification
        
        # 通路 B: 原生信号通路 (ResNet/CNN 风格)
        # 我们让这一层的输出维度也等于 d_model，方便融合
        self.raw_path = RawSignalPath(input_channels=1, out_dim=d_model)
        
        # 融合层 (Fusion Head)
        # 输入是两个通道拼接：d_model + d_model = 2 * d_model
        self.fusion_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # 1. 基础维度修正 [B, L] -> [B, 1, L]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # 2. 对齐预训练长度 (1000点)
        if x.shape[-1] != self.target_points:
            x_aligned = F.interpolate(x, size=(self.target_points,), mode='linear', align_corners=False)
        else:
            x_aligned = x

        # --- 通路 A: 预训练特征提取 ---
        # 必须带上 Z-Score 归一化
        x_norm = (x_aligned - x_aligned.mean(dim=-1, keepdim=True)) / (x_aligned.std(dim=-1, keepdim=True) + 1e-8)
        with torch.no_grad(): # 如果你选择冻结 Encoder
            feat_pretrained, _ = self.encoder(x_norm) # [B, d_model]
        
        # --- 通路 B: 原生特征提取 ---
        # 原生通路可以直接使用对齐后的信号，甚至不需要 Z-Score（让模型自己学）
        feat_raw = self.raw_path(x_aligned) # [B, d_model]
        
        # --- 特征融合 ---
        # 拼接特征: [B, d_model * 2]
        combined_feat = torch.cat([feat_pretrained, feat_raw], dim=-1)
        
        # 最终预测
        out = self.fusion_head(combined_feat)
        return out

class Exp_Finetune:
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.early_stop_lr = config.early_stop_lr
        self.epochs = config.epochs
        self.lead = config.lead
        self.model_type = config.model_type
        self.run_id = config.dataset
        self.flag = True
        self.device = torch.device(f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.dataset = config.dataset

        self.d_model = config.d_model

        self.encoder_ckpt_path = '/public/home/ai_user_1/DC/hcy/checkpoints/mimic/mimic_99_encoder.pth'

    def _filter_encoder_state_dict(self, ckpt: dict):
        # 过滤掉下游头相关键，确保只加载 encoder 部分
        bad_prefix = ('linear.', 'regressor.', 'head.', 'pooling.', 'adapter.', 'attn_pool.', 'pool_norm.')
        return {k: v for k, v in ckpt.items() if not k.startswith(bad_prefix)}

    def build_model(self):
        # 1) 初始化预训练 Encoder
        encoder = model_builder.PPGPatchEncoder(
            num_leads=self.lead, 
            d_model=self.d_model, 
            nheads=4, # 必须与预训练一致
            num_layers=3
        ).to(self.device)

        # 2) 加载预训练权重
        print(f"📥 加载预训练权重: {self.encoder_ckpt_path}")
        if self.dataset in ['fangchan', 'huxipinlu']:
            encoder_ckpt_path = '/public/home/ai_user_1/DC/hcy/checkpoints/tp1000_td30/mimic_99_encoder.pth'
        elif self.dataset in ['xintiao','xueya']:
            encoder_ckpt_path = '/public/home/ai_user_1/DC/hcy/checkpoints/tp1200_td30/mimic_99_encoder.pth'
        else:
            print(f"📥 加载预训练权重失败")
            
            
        ckpt = torch.load(encoder_ckpt_path, map_location='cpu')
        new_state_dict = {k.replace('ppg_encoder.', ''): v for k, v in ckpt.items()}
        encoder.load_state_dict(new_state_dict, strict=False)

        # 3) 构建双通路模型
        is_classification = (self.dataset == "fangchan")
        model = PPGDualPathModel(
            pretrained_encoder=encoder,
            d_model=self.d_model,
            target_points=1000,
            is_classification=is_classification
        ).to(self.device)

        # 4) 冻结通路 A (可选，建议初期冻结)
        for p in model.encoder.parameters():
            p.requires_grad = False
            
        # 确保通路 B 和融合头是开启梯度的
        for p in model.raw_path.parameters():
            p.requires_grad = True
        for p in model.fusion_head.parameters():
            p.requires_grad = True

        return model

    def _get_trainable_params(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise RuntimeError("No trainable parameters found. Check whether head/adapter are set to requires_grad=True.")
        return params

    def finetune(self):
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score

        saved_dir = f'/public/home/ai_user_1/DC/hcy/checkpoints/{self.run_id}_{self.lead}_lead_{self.model_type}'
        os.makedirs(saved_dir, exist_ok=True)

        is_classification = (self.dataset == "fangchan")

        if self.dataset in ["fangchan", "huxipinlu", "xintiao", "xueya"]:
            dataset_path = f'/public/home/ai_user_1/DC/hcy/dataset/down_steam_dataset/{self.dataset}'
        else:
            print(f"[WARN] 未知数据集: {self.dataset}")
            return

        cv_loaders, test_loader = downstream_dataset.load_dataloaders(dataset_path, batch_size=self.batch_size)

        if is_classification:
            base_criterion = nn.BCEWithLogitsLoss()
            plateau_mode = 'max'
        else:
            base_criterion = nn.MSELoss()
            plateau_mode = 'min'

        step = 0

        def _to_binary01(y):
            y = y.astype(float).reshape(-1)
            if set(np.unique(y)).issubset({-1.0, 1.0}):
                y = (y + 1.0) / 2.0
            if set(np.unique(y)).issubset({1.0, 2.0}):
                y = y - 1.0
            y = (y > 0).astype(int)
            return y

        def eval_regression(model, dataloader):
            model.eval()
            all_gt, all_pred = [], []
            with torch.no_grad():
                for batch in dataloader:
                    input_x, input_y = tuple(t.to(self.device) for t in batch)
                    pred = model(input_x)  # [B,1]
                    all_pred.append(pred.squeeze(-1).cpu().numpy())
                    all_gt.append(input_y.view(-1).cpu().numpy())
            y_true = np.concatenate(all_gt, axis=0)
            y_pred = np.concatenate(all_pred, axis=0)
            
            save_path = 'DC/hcy/PPG_Clip/npy/'
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # 1) 推荐：npz
                # 你之后 load：data=np.load(path); y_true=data["y_true"]; y_pred=data["y_pred"]
                if not save_path.endswith(".npz"):
                    save_path_npz = save_path + ".npz"
                else:
                    save_path_npz = save_path
                np.savez(save_path_npz, y_true=y_true, y_pred=y_pred)
            
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = float(np.sqrt(mse))
            r2 = r2_score(y_true, y_pred)
            return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}

        def find_best_threshold(y_true, y_prob):
            uniq = np.unique(y_prob)
            if uniq.size > 512:
                idx = np.linspace(0, uniq.size - 1, 512).astype(int)
                uniq = uniq[idx]
            cands = np.concatenate(([0.0], uniq, [1.0]))
            best_f1, best_th = 0.0, 0.5
            for th in cands:
                y_pred = (y_prob >= th).astype(int)
                tp = ((y_true == 1) & (y_pred == 1)).sum()
                fp = ((y_true == 0) & (y_pred == 1)).sum()
                fn = ((y_true == 1) & (y_pred == 0)).sum()
                denom = (2 * tp + fp + fn)
                f1 = (2 * tp) / denom if denom > 0 else 0.0
                if f1 > best_f1:
                    best_f1, best_th = f1, th
            return best_th, best_f1

        def eval_classification(model, dataloader, threshold=None):
            model.eval()
            all_gt, all_logit = [], []
            with torch.no_grad():
                for batch in dataloader:
                    input_x, input_y = tuple(t.to(self.device) for t in batch)
                    logit = model(input_x)  # [B,1]
                    all_logit.append(logit.squeeze(-1).cpu().numpy())
                    all_gt.append(input_y.view(-1).cpu().numpy())

            y_true = np.concatenate(all_gt, axis=0).astype(int)
            y_true = _to_binary01(y_true)
            y_logit = np.concatenate(all_logit, axis=0)
            y_prob = 1.0 / (1.0 + np.exp(-y_logit))

            if threshold is None:
                best_th, best_f1 = find_best_threshold(y_true, y_prob)
            else:
                best_th = threshold
                y_pred_tmp = (y_prob >= best_th).astype(int)
                try:
                    best_f1 = f1_score(y_true, y_pred_tmp)
                except:
                    best_f1 = 0.0

            y_pred = (y_prob >= best_th).astype(int)
            try:
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
            except:
                precision, recall = 0.0, 0.0
            acc = accuracy_score(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, y_prob)
            except:
                auc = float('nan')
            try:
                ap = average_precision_score(y_true, y_prob)
            except:
                ap = float('nan')

            return {
                "f1": float(best_f1), "precision": float(precision), "recall": float(recall),
                "acc": float(acc), "auc": float(auc), "ap": float(ap), "threshold": float(best_th)
            }

        for i, fold in enumerate(cv_loaders):
            print(f"\n🌀 [Fold {i+1}/{len(cv_loaders)}] 开始训练...")

            train_loader = fold['train']
            val_loader = fold['val']

            # 每个 fold 重新初始化模型，避免 CV 泄漏
            model = self.build_model()

            # 每个 fold 重新建优化器/调度器
            trainable_params = self._get_trainable_params(model)
            optimizer = optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, mode=plateau_mode)

            # 每 fold 独立最佳指标
            if is_classification:
                best_val_metric = -float('inf')
            else:
                best_val_metric = float('inf')

            # 分类任务：每 fold 动态计算 pos_weight，并替换 criterion
            criterion = base_criterion
            if is_classification:
                pos_cnt, neg_cnt = 0, 0
                for batch in train_loader:
                    _, yb = batch
                    yb = yb.view(-1).cpu().numpy()
                    yb = _to_binary01(yb)
                    pos_cnt += int((yb == 1).sum())
                    neg_cnt += int((yb == 0).sum())
                eps = 1e-6
                pos_weight_value = float(neg_cnt + eps) / float(pos_cnt + eps)
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=self.device))
                total = pos_cnt + neg_cnt
                pos_rate = pos_cnt / max(1, total)
                print(f"[Info] Class balance -> pos={pos_cnt}, neg={neg_cnt}, pos_rate={pos_rate:.4f}, pos_weight={pos_weight_value:.3f}")

            for epoch in range(self.epochs):
                model.train()
                total_loss = 0.0

                for batch in train_loader:
                    input_x, input_y = tuple(t.to(self.device) for t in batch)

                    out = model(input_x).squeeze(-1)  # [B]

                    if is_classification:
                        y_np = input_y.detach().view(-1).cpu().numpy()
                        y_np = _to_binary01(y_np)
                        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)
                        loss = criterion(out, y)
                    else:
                        y = input_y.float().view(-1)
                        loss = criterion(out, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    step += 1
                    total_loss += float(loss.item())

                avg_train_loss = total_loss / max(1, len(train_loader))

                # 评估
                if is_classification:
                    val_metrics = eval_classification(model, val_loader, threshold=None)
                    best_th = val_metrics["threshold"]
                    test_metrics = eval_classification(model, test_loader, threshold=best_th)
                    current_metric = val_metrics["f1"]
                else:
                    val_metrics = eval_regression(model, val_loader)
                    test_metrics = eval_regression(model, test_loader)
                    current_metric = val_metrics["rmse"]

                print(f"\n📘 Fold {i+1} | Epoch {epoch+1}/{self.epochs}")
                print(f"🔹 Train Loss: {avg_train_loss:.6f}")

                if is_classification:
                    print(f"🔸 Val   -> F1: {val_metrics['f1']:.4f} | P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f} | "
                          f"Acc: {val_metrics['acc']:.4f} | AUC: {val_metrics['auc']:.4f} | AP: {val_metrics['ap']:.4f} | Thr*: {val_metrics['threshold']:.3f}")
                    print(f"🔸 Test  -> F1: {test_metrics['f1']:.4f} | P: {test_metrics['precision']:.4f} | R: {test_metrics['recall']:.4f} | "
                          f"Acc: {test_metrics['acc']:.4f} | AUC: {test_metrics['auc']:.4f} | AP: {test_metrics['ap']:.4f} | Thr = {val_metrics['threshold']:.3f}")
                else:
                    print(f"🔸 Val   -> MSE: {val_metrics['mse']:.4f} | MAE: {val_metrics['mae']:.4f} | RMSE: {val_metrics['rmse']:.4f} | R²: {val_metrics['r2']:.4f}")
                    print(f"🔸 Test  -> MSE: {test_metrics['mse']:.4f} | MAE: {test_metrics['mae']:.4f} | RMSE: {test_metrics['rmse']:.4f} | R²: {test_metrics['r2']:.4f}")

                # 保存最佳
                improved = (current_metric > best_val_metric) if is_classification else (current_metric < best_val_metric)
                if improved:
                    best_val_metric = current_metric
                    unified_val_rmse = (val_metrics["rmse"] if not is_classification else (1.0 - val_metrics["f1"]))

                    print("💾 新最佳模型已保存！📍")
                    util.save_checkpoint({
                        'epoch': epoch,
                        'step': step,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'val_rmse': unified_val_rmse,
                        'val_metric': best_val_metric,
                        'task': 'classification' if is_classification else 'regression',
                        'd_model': self.d_model,
                        'lead': self.lead,
                    }, saved_dir)

                scheduler.step(current_metric)

            if is_classification:
                print(f"\n✅ [Fold {i+1}] 完成训练。最优 Val F1: {best_val_metric:.4f}")
            else:
                print(f"\n✅ [Fold {i+1}] 完成训练。最优 Val RMSE: {best_val_metric:.4f}")

        print("\n" + "=" * 60)
        print("🎉 所有 Fold 训练完成！全部任务顺利结束 🎯")
        print("=" * 60)
