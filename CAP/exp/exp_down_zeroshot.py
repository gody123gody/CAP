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
        self.device = torch.device(f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')

    def evaluate_zero_shot(self):
        # 创建保存路径
        saved_dir = f'/gemini/platform/public/aigc/Lirui/chengding/hcy/checkpoints/{self.run_id}_{self.lead}_lead_{self.model_type}'
        os.makedirs(saved_dir, exist_ok=True)

        # 加载测试集（不需要交叉验证）
        dataset_path = '/gemini/platform/public/aigc/Lirui/chengding/val_dataset/fangchan'
        _, test_loader = downstream_dataset.load_dataloaders(dataset_path, batch_size=64)

        # 加载模型
        model = transformer.PPGTransformerEncoder(
            input_dim=self.lead,
            d_model=self.d_model,
            nhead=4,
            num_layers=3,
            flag=self.flag
        ).to(self.device)

        # 加载预训练权重（跳过linear）
        checkpoint = torch.load('/gemini/platform/public/aigc/Lirui/chengding/hcy/checkpoints/1_lead_model_50_encoder.pth', map_location=self.device)
        state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('linear.')}
        model.load_state_dict(state_dict, strict=False)

        # 重建线性层
        model.regressor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 1)
        ).to(self.device)

        # 这里我在思考 因为这个线性层是完全随机的 可能这样意义不大 要不要就是利用预训练部分得到的特征值做一个平均值或者什么的来直接给出预测值

        # 设置评估模式
        model.eval()
        all_gt, all_pred = [], []

        print("\n🚀 正在进行 Zero-Shot 推理评估...")

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_x, input_y = tuple(t.to(self.device) for t in batch)
                pred = model(input_x)
                all_pred.append(pred.cpu().numpy())
                all_gt.append(input_y.view(-1, 1).cpu().numpy())

        all_gt = np.concatenate(all_gt, axis=0)
        all_pred = np.concatenate(all_pred, axis=0)

        # 评估指标
        test_mse = mean_squared_error(all_gt, all_pred)
        test_mae = mean_absolute_error(all_gt, all_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(all_gt, all_pred)

        print("\n🎯 Zero-Shot 评估结果:")
        print(f"🔹 MSE: {test_mse:.4f}")
        print(f"🔹 MAE: {test_mae:.4f}")
        print(f"🔹 RMSE: {test_rmse:.4f}")
        print(f"🔹 R²: {test_r2:.4f}")


