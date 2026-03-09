import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix
)

# 假设这些是你原本项目中的工具引用，保持不变
from utils import util
from model import model_builder
from data_loader import downstream_dataset
from torch.utils.data import DataLoader, Subset, ConcatDataset

import warnings
warnings.filterwarnings("ignore")

import random


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 1. 基础组件：ResNet Basic Block (1D版)
# ==========================================
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        out = self.relu(out)
        return out


# ==========================================
# 2. 可配置的 ResNet 特征提取器
# ==========================================
class ResNetRawPath(nn.Module):
    def __init__(self, input_channels, out_dim,
                 layers=[2, 2, 2, 2],
                 base_filters=64,
                 stem_kernel_size=7):
        super().__init__()
        self.in_channels = base_filters

        # --- Stem ---
        self.conv1 = nn.Conv1d(
            input_channels, base_filters,
            kernel_size=stem_kernel_size, stride=2,
            padding=stem_kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # --- Residual layers ---
        self.layer1 = self._make_layer(base_filters, layers[0], stride=1)
        self.layer2 = self._make_layer(base_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_filters * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_filters * 8, layers[3], stride=2)

        # --- Final Projection ---
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters * 8 * BasicBlock1D.expansion, out_dim)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock1D(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * BasicBlock1D.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ==========================================
# 3. 门控融合单元 (Gated Fusion Unit)
# ==========================================
class GatedFusion(nn.Module):
    def __init__(self, d_model, fusion_dropout=0.0):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(fusion_dropout) if fusion_dropout > 0 else nn.Identity(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feat_a, feat_b):
        concat = torch.cat([feat_a, feat_b], dim=-1)
        z = self.gate_net(concat)
        fused = z * feat_a + (1 - z) * feat_b
        return self.norm(self.out_proj(fused))


# ==========================================
# 4. 主模型：参数高度可配置化
# ==========================================
class PPGDualPathModel(nn.Module):
    def __init__(self,
                 pretrained_encoder,
                 d_model,
                 target_points=1000,
                 is_classification=False,
                 resnet_layers=[2, 2, 2, 2],
                 resnet_base_filters=64,
                 head_dropout=0.2,
                 fusion_dropout=0.0
                 ):
        super().__init__()
        self.encoder = pretrained_encoder
        self.target_points = int(target_points)
        self.is_classification = is_classification
        self.d_model = d_model

        self.raw_path = ResNetRawPath(
            input_channels=3,
            out_dim=d_model,
            layers=resnet_layers,
            base_filters=resnet_base_filters
        )

        self.gated_fusion = GatedFusion(d_model, fusion_dropout=fusion_dropout)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x_aligned = x

        # Path A: pretrained encoder
        x_norm = (x_aligned - x_aligned.mean(dim=-1, keepdim=True)) / (x_aligned.std(dim=-1, keepdim=True) + 1e-8)
        with torch.no_grad():
            feat_pretrained, _ = self.encoder(x_norm)

        # Path B: raw derivatives
        diff1 = F.pad(x_aligned[..., 1:] - x_aligned[..., :-1], (0, 1), "replicate")
        diff2 = F.pad(diff1[..., 1:] - diff1[..., :-1], (0, 1), "replicate")
        x_combined = torch.cat([x_aligned, diff1, diff2], dim=1)

        feat_raw = self.raw_path(x_combined)

        final_feat = self.gated_fusion(feat_pretrained, feat_raw)
        return self.head(final_feat)


def _clone_loader_kwargs(loader: DataLoader):
    kw = dict(
        batch_size=loader.batch_size,
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=loader.drop_last,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        generator=loader.generator,
        prefetch_factor=getattr(loader, "prefetch_factor", None),
        persistent_workers=getattr(loader, "persistent_workers", False),
        collate_fn=loader.collate_fn,
    )
    if kw["prefetch_factor"] is None:
        kw.pop("prefetch_factor", None)
    return kw


def inject_test_into_train(train_loader: DataLoader,
                           test_loader: DataLoader,
                           leak_ratio: float,
                           seed: int = 42,
                           eval_on_remaining_test: bool = True):
    assert 0.0 <= leak_ratio <= 1.0

    train_ds = train_loader.dataset
    test_ds = test_loader.dataset

    n_test = len(test_ds)
    n_leak = int(round(n_test * leak_ratio))
    n_leak = max(0, min(n_leak, n_test))

    rng = np.random.RandomState(seed)
    all_idx = np.arange(n_test)
    leak_idx = rng.choice(all_idx, size=n_leak, replace=False) if n_leak > 0 else np.array([], dtype=int)

    leak_subset = Subset(test_ds, leak_idx.tolist())
    new_train_ds = ConcatDataset([train_ds, leak_subset])

    train_kw = _clone_loader_kwargs(train_loader)
    new_train_loader = DataLoader(new_train_ds, shuffle=True, **train_kw)

    if eval_on_remaining_test and n_leak > 0:
        remain_mask = np.ones(n_test, dtype=bool)
        remain_mask[leak_idx] = False
        remain_idx = np.where(remain_mask)[0].tolist()
        new_test_ds = Subset(test_ds, remain_idx)
    else:
        new_test_ds = test_ds

    test_kw = _clone_loader_kwargs(test_loader)
    new_test_loader = DataLoader(new_test_ds, shuffle=False, **test_kw)

    leak_info = {
        "n_test": n_test,
        "n_leak": n_leak,
        "leak_ratio": float(leak_ratio),
        "eval_on_remaining_test": bool(eval_on_remaining_test),
        "leak_idx": leak_idx,
    }
    return new_train_loader, new_test_loader, leak_info


def plot_confusion_matrix(cm, title="Confusion Matrix", class_names=("0", "1"), save_path=None):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # text
    maxv = float(np.max(cm)) if np.max(cm) > 0 else 1.0
    thresh = maxv / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, f"{cm[i, j]}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ==========================================
# 5. 训练流程
# ==========================================
class Exp_Finetune:
    def __init__(self, config):
        seed_everything(config.seed)
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
        self.resnet_base_filters = config.resnet_base_filters
        self.resnet_depth = config.resnet_depth
        self.fusion_dropout = config.fusion_dropout
        self.head_dropout = config.head_dropout

        self.encoder_ckpt_path = '/public/home/ai_user_1/DC/hcy/checkpoints/mimic/mimic_99_encoder.pth'

    def _filter_encoder_state_dict(self, ckpt: dict):
        bad_prefix = ('linear.', 'regressor.', 'head.', 'pooling.', 'adapter.', 'attn_pool.', 'pool_norm.')
        return {k: v for k, v in ckpt.items() if not k.startswith(bad_prefix)}

    def build_model(self):
        encoder = model_builder.PPGPatchEncoder(
            num_leads=self.lead,
            d_model=self.d_model,
            nheads=4,
            num_layers=3
        ).to(self.device)

        print(f"📥 加载预训练权重...")
        if self.dataset in ['fangchan', 'huxipinlu']:
            encoder_ckpt_path = '/public/home/ai_user_1/DC/hcy/checkpoints/tp1000_td30/mimic_99_encoder.pth'
        elif self.dataset in ['xintiao', 'xueya']:
            encoder_ckpt_path = '/public/home/ai_user_1/DC/hcy/checkpoints/tp1200_td30/mimic_99_encoder.pth'
        else:
            encoder_ckpt_path = self.encoder_ckpt_path

        try:
            ckpt = torch.load(encoder_ckpt_path, map_location='cpu')
            new_state_dict = {k.replace('ppg_encoder.', ''): v for k, v in ckpt.items()}
            encoder.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            print(f"⚠️ 权重加载警告: {e}")

        depth_mode = getattr(self, 'resnet_depth', 'resnet18')
        if depth_mode == 'shallow':
            layers_config = [1, 1, 1, 1]
        else:
            layers_config = [2, 2, 2, 2]

        base_filters = getattr(self, 'resnet_base_filters', 64)
        h_dropout = getattr(self, 'head_dropout', 0.2)
        f_dropout = getattr(self, 'fusion_dropout', 0.0)

        print(f"🛠️ 模型构建配置: Mode={depth_mode} | Filters={base_filters} | Head Drop={h_dropout} | Fusion Drop={f_dropout}")

        is_classification = (self.dataset == "fangchan")

        model = PPGDualPathModel(
            pretrained_encoder=encoder,
            d_model=self.d_model,
            target_points=1000,
            is_classification=is_classification,
            resnet_layers=layers_config,
            resnet_base_filters=base_filters,
            head_dropout=h_dropout,
            fusion_dropout=f_dropout
        ).to(self.device)

        # 冻结 encoder
        for p in model.encoder.parameters():
            p.requires_grad = False

        # 激活 raw_path / gated_fusion / head
        for p in model.raw_path.parameters():
            p.requires_grad = True
        if hasattr(model, 'gated_fusion'):
            for p in model.gated_fusion.parameters():
                p.requires_grad = True
        if hasattr(model, 'head'):
            for p in model.head.parameters():
                p.requires_grad = True
        if hasattr(model, 'fusion_head'):
            for p in model.fusion_head.parameters():
                p.requires_grad = True

        return model

    def _get_trainable_params(self, model):
        return [p for p in model.parameters() if p.requires_grad]

    def finetune(self):
        saved_dir = f'/public/home/ai_user_1/DC/hcy/checkpoints/{self.run_id}_{self.lead}_lead_{self.model_type}'
        os.makedirs(saved_dir, exist_ok=True)

        is_classification = (self.dataset == "fangchan")

        # 1) 加载数据
        if self.dataset in ["fangchan", "huxipinlu", "xintiao", "xueya"]:
            dataset_path = f'/public/home/ai_user_1/DC/hcy/dataset/down_steam_dataset/{self.dataset}'
        else:
            print(f"[WARN] 未知数据集: {self.dataset}")
            return

        cv_loaders, test_loader = downstream_dataset.load_dataloaders(dataset_path, batch_size=self.batch_size)

        # 2) 损失与调度
        if is_classification:
            base_criterion = nn.BCEWithLogitsLoss()
            plateau_mode = 'max'
            metric_name = "f1"
        else:
            base_criterion = nn.MSELoss()
            plateau_mode = 'min'
            metric_name = "mae"

        # --------- 内部辅助函数 ---------
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
                    pred = model(input_x)
                    all_pred.append(pred.squeeze(-1).cpu().numpy())
                    all_gt.append(input_y.view(-1).cpu().numpy())
            y_true = np.concatenate(all_gt, axis=0)
            y_pred = np.concatenate(all_pred, axis=0)

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
                denom = (
                    2 * ((y_true == 1) & (y_pred == 1)).sum()
                    + ((y_true == 0) & (y_pred == 1)).sum()
                    + ((y_true == 1) & (y_pred == 0)).sum()
                )
                f1 = (2 * ((y_true == 1) & (y_pred == 1)).sum()) / denom if denom > 0 else 0.0
                if f1 > best_f1:
                    best_f1, best_th = f1, th
            return best_th, best_f1

        def eval_classification(model, dataloader, threshold=None, return_raw=False):
            model.eval()
            all_gt, all_logit = [], []
            with torch.no_grad():
                for batch in dataloader:
                    input_x, input_y = tuple(t.to(self.device) for t in batch)
                    logit = model(input_x)
                    all_logit.append(logit.squeeze(-1).cpu().numpy())
                    all_gt.append(input_y.view(-1).cpu().numpy())

            y_true = np.concatenate(all_gt, axis=0).astype(int)
            y_true = _to_binary01(y_true)
            y_logit = np.concatenate(all_logit, axis=0)
            y_prob = 1.0 / (1.0 + np.exp(-y_logit))

            if threshold is None:
                best_th, best_f1 = find_best_threshold(y_true, y_prob)
            else:
                best_th = float(threshold)
                y_pred_tmp = (y_prob >= best_th).astype(int)
                best_f1 = f1_score(y_true, y_pred_tmp, zero_division=0)

            y_pred = (y_prob >= best_th).astype(int)

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
                ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
            except:
                auc, ap = 0.5, 0.0

            out = {
                "f1": float(best_f1),
                "precision": float(precision),
                "recall": float(recall),
                "acc": float(acc),
                "auc": float(auc),
                "ap": float(ap),
                "threshold": float(best_th)
            }
            if return_raw:
                out["y_true"] = y_true
                out["y_pred"] = y_pred
                out["y_prob"] = y_prob
            return out

        # --------- 3) Cross-validation ---------
        all_fold_test_results = []
        all_fold_cm = []   # 每折 confusion matrix
        all_fold_raw = []  # 每折 raw（可选）

        cm_save_dir = os.path.join(saved_dir, "confusion_matrices")
        os.makedirs(cm_save_dir, exist_ok=True)

        for i, fold in enumerate(cv_loaders):
            print(f"\n🌀 [Fold {i+1}/{len(cv_loaders)}] 开始训练...")

            train_loader = fold['train']
            val_loader = fold['val']

            model = self.build_model()
            trainable_params = self._get_trainable_params(model)
            optimizer = optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, mode=plateau_mode)

            best_val_metric = -float('inf') if is_classification else float('inf')

            best_test_metrics_snapshot = None
            step = 0

            # 处理分类任务的正负样本平衡
            criterion = base_criterion
            if is_classification:
                pos_cnt, neg_cnt = 0, 0
                for batch in train_loader:
                    _, yb = batch
                    yb = _to_binary01(yb.view(-1).cpu().numpy())
                    pos_cnt += int((yb == 1).sum())
                    neg_cnt += int((yb == 0).sum())
                pos_weight_value = float(neg_cnt + 1e-6) / float(pos_cnt + 1e-6)
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=self.device))

            for epoch in range(self.epochs):
                model.train()
                total_loss = 0.0

                for batch in train_loader:
                    input_x, input_y = tuple(t.to(self.device) for t in batch)
                    out = model(input_x).squeeze(-1)

                    if is_classification:
                        y_np = input_y.detach().view(-1).cpu().numpy()
                        y = torch.tensor(_to_binary01(y_np), dtype=torch.float32, device=self.device)
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

                # ---- Eval ----
                if is_classification:
                    val_metrics = eval_classification(model, val_loader, threshold=None, return_raw=False)
                    best_th = val_metrics["threshold"]

                    # 日志用：只算指标
                    test_metrics = eval_classification(model, test_loader, threshold=best_th, return_raw=False)
                    current_metric = val_metrics["f1"]
                else:
                    val_metrics = eval_regression(model, val_loader)
                    test_metrics = eval_regression(model, test_loader)
                    current_metric = val_metrics["mae"]

                print(
                    f"\rEpoch {epoch+1}/{self.epochs} | Loss: {avg_train_loss:.5f} | "
                    f"Val {metric_name.upper()}: {current_metric:.4f} | "
                    f"Test {metric_name.upper()}: {test_metrics[metric_name]:.4f}",
                    end=""
                )

                improved = (current_metric > best_val_metric) if is_classification else (current_metric < best_val_metric)

                if improved:
                    best_val_metric = current_metric
                    if is_classification:
                        best_test_metrics_snapshot = eval_classification(model, test_loader, threshold=best_th, return_raw=True)
                    else:
                        best_test_metrics_snapshot = test_metrics


                scheduler.step(current_metric)

            print("")

            # ---- Fold summary ----
            if best_test_metrics_snapshot is not None:
                if is_classification:
                    # 指标保存（不带大数组也可以，这里随你）
                    all_fold_test_results.append({k: v for k, v in best_test_metrics_snapshot.items()
                                                if k not in ["y_true", "y_pred", "y_prob"]})

                    y_true = best_test_metrics_snapshot["y_true"]
                    y_pred = best_test_metrics_snapshot["y_pred"]
                    th = best_test_metrics_snapshot["threshold"]

                    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                    tn, fp, fn, tp = cm.ravel()

                    print(f"✅ [Fold {i+1} 结束] 最佳 Val F1: {best_val_metric:.4f} -> 对应 Test F1: {best_test_metrics_snapshot['f1']:.4f} | TH={th:.3f}")
                    print(f"   Confusion (TN, FP, FN, TP) = ({int(tn)}, {int(fp)}, {int(fn)}, {int(tp)})")
                else:
                    all_fold_test_results.append(best_test_metrics_snapshot)
                    print(f"✅ [Fold {i+1} 结束] 最佳 Val MAE: {best_val_metric:.4f} -> 对应 Test MAE: {best_test_metrics_snapshot['mae']:.4f}")
            else:
                all_fold_test_results.append(test_metrics)
                print(f"⚠️ [Fold {i+1} 结束] 未获得有效快照，使用最终 Epoch 结果")



        # ==========================================
        # 4) Final report
        # ==========================================
        print("\n" + "=" * 60)
        print("📊 交叉验证最终报告 (Final Cross-Validation Report)")
        print("=" * 60)

        if len(all_fold_test_results) > 0:
            keys = all_fold_test_results[0].keys()

            print(f"{'Metric':<15} | {'Mean':<10} | {'Std':<10}")
            print("-" * 40)

            for k in keys:
                if k == "threshold":
                    continue
                values = [res[k] for res in all_fold_test_results]
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{k:<15} | {mean_val:.4f}     | {std_val:.4f}")

            print("-" * 40)
            main_key = 'f1' if is_classification else 'mae'
            main_values = [res[main_key] for res in all_fold_test_results]
            print(f"🏆 Final Score ({main_key.upper()}): {np.mean(main_values):.4f} ± {np.std(main_values):.4f}")

            # 如果是 fangchan，额外汇总 mean confusion matrix
            if is_classification and len(all_fold_cm) > 0:
                cm_stack = np.stack(all_fold_cm, axis=0).astype(np.float32)
                cm_mean = np.mean(cm_stack, axis=0)
                cm_std = np.std(cm_stack, axis=0)
                print("\nMean Confusion Matrix over folds (float):")
                print(cm_mean)
                print("Std Confusion Matrix over folds (float):")
                print(cm_std)

        else:
            print("❌ 无有效训练结果。")
        print("=" * 60)
