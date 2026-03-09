import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import util
import torch.nn as nn
from model import transformer
from data_loader import downstream_dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # 过滤AMP弃用警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="Glyph.*missing from font")

class Exp_Finetune:
    def __init__(self,config):
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.early_stop_lr = config.early_stop_lr
        self.epochs = config.epochs
        self.lead = config.lead
        self.model_type = config.model_type
        self.d_model = config.d_model
        self.run_id = config.dataset
        self.flag = True
        self.few_ratio = 0.05
        self.device = torch.device(f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')

    def finetune(self):
        
        saved_dir = f'/gemini/platform/public/aigc/Lirui/chengding/hcy/checkpoints/{self.run_id}_{self.lead}_lead_{self.model_type}'

        if self.dataset in ["fangchan", "huxipinlu", "xintiao", "xueya"]:
            dataset_path = '/gemini/platform/public/aigc/Lirui/chengding/val_dataset/'+self.dataset
        else:
            return
        cv_loaders, test_loader = downstream_dataset.load_dataloaders(dataset_path, batch_size=self.batch_size)

        model = transformer.PPGTransformerEncoder(
            input_dim=self.lead,
            d_model=256,
            nhead=4,
            num_layers=3,
            flag=self.flag
        ).to(self.device)

        # 加载预训练参数（排除 linear）
        checkpoint = torch.load('/gemini/platform/public/aigc/Lirui/chengding/hcy/checkpoints/1_lead_model_50_encoder.pth', map_location=self.device)
        state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('linear.')}
        model.load_state_dict(state_dict, strict=False)

        # 重新定义 linear 层
        model.regressor = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Linear(self.d_model // 2, 1)
            ).to(self.device)
        # 优化器 & 损失
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, mode='max')

        step = 0
        loss_list = []
        best_val_rmse = float('inf') 

        for i, fold in enumerate(cv_loaders):
            print(f"\n🌀 [Fold {i+1}/{len(cv_loaders)}] 开始训练...")

            # train_loader = fold['train']
            train_loader_full = fold['train']

            # Few-shot: 设定训练样本比例
            few_shot_ratio = self.few_ratio  # 使用5%的训练数据
            few_shot_samples = int(len(train_loader_full.dataset) * few_shot_ratio)

            # 从完整训练集随机采样 few-shot 数据
            few_shot_subset = torch.utils.data.Subset(train_loader_full.dataset, np.random.choice(len(train_loader_full.dataset), few_shot_samples, replace=False))

            # 创建新的 train_loader
            train_loader = torch.utils.data.DataLoader(few_shot_subset, batch_size=self.batch_size, shuffle=True)

            val_loader = fold['val']

            for epoch in range(self.epochs):
                def eval_model(dataloader, name="Val"):
                    model.eval()
                    all_gt, all_pred = [], []
                    with torch.no_grad():
                        for batch in dataloader:
                            input_x, input_y = tuple(t.to(self.device) for t in batch)
                            pred = model(input_x)  # [B, 1]
                            all_pred.append(pred.cpu().numpy())  # [B, 1]
                            all_gt.append(input_y.view(-1, 1).cpu().numpy())  # [B, 1]
                    all_gt = np.concatenate(all_gt, axis=0)      # [N, 1]
                    all_pred = np.concatenate(all_pred, axis=0)  # [N, 1]
                    return all_gt, all_pred
                model.train()
                total_loss = 0

                for batch in train_loader:
                    input_x, input_y = tuple(t.to(self.device) for t in batch)
                    outputs = model(input_x)
                    loss = criterion(outputs, input_y.unsqueeze(-1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1
                    total_loss += loss.item()

                avg_train_loss = total_loss / len(train_loader)
                loss_list.append(avg_train_loss)

                # ========= 评估 ========= #
                if epoch % 1 == 0:
                    val_gt, val_pred = eval_model(val_loader, "Val")
                    val_mse = mean_squared_error(val_gt, val_pred)
                    val_mae = mean_absolute_error(val_gt, val_pred)
                    val_rmse = np.sqrt(mean_squared_error(val_gt, val_pred))
                    val_r2 = r2_score(val_gt, val_pred)

                    test_gt, test_pred = eval_model(test_loader, "Test")
                    test_mse = mean_squared_error(test_gt, test_pred)
                    test_mae = mean_absolute_error(test_gt, test_pred)
                    test_rmse = np.sqrt(mean_squared_error(test_gt, test_pred))
                    test_r2 = r2_score(test_gt, test_pred)

                    print(f"\n📘 Fold {i+1} | Epoch {epoch+1}/{self.epochs}")
                    print(f"🔹 Train Loss: {avg_train_loss:.4f}")
                    print(f"🔸 Val   -> MSE: {val_mse:.4f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | R²: {val_r2:.4f}")
                    print(f"🔸 Test  -> MSE: {test_mse:.4f} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")

                    # 保存最佳模型（以最小 Val RMSE 为准）
                    if val_rmse < best_val_rmse:
                        best_val_rmse = val_rmse
                        print("💾 新最佳模型已保存！📍")
                        util.save_checkpoint({
                            'epoch': epoch,
                            'step': step,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'val_rmse': val_rmse,
                        }, saved_dir)

                    scheduler.step(val_rmse)

            print(f"\n✅ [Fold {i+1}] 完成训练。最优 Val RMSE: {best_val_rmse:.4f}")

        print("\n" + "=" * 60)
        print("🎉 所有 Fold 训练完成！全部任务顺利结束 🎯")
        print("=" * 60)

