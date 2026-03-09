import random

import torch.distributed as dist
import os
import numpy as np
import argparse

from exp import exp_trainer
from data_loader import pretrain_dataset
from model import model_builder
from exp import exp_down_fullshot
from exp import exp_down_fewshot
from exp import exp_down_zeroshot

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def main():

    parser = argparse.ArgumentParser(description='Finetune')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight_decay', type=int, default=1.0e-8, help='weight_decay')
    parser.add_argument('--early_stop_lr', type=int, default=1e-5, help='early_stop_lr')
    parser.add_argument('--epochs', type=int, default=15, help='epochs')
    parser.add_argument('--lead', type=int, default=1, help='lead count')
    parser.add_argument('--model_type', type=str, default='Linear', help='[Linear, KNN]')
    parser.add_argument('--d_model', type=int, default=256, help='d model')
    parser.add_argument('--dataset', type=str, default='huxipinlu', help='[fangchan, huxipinlu, xintiao, xueya]')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--seed', type=int, default=2021, help='GPU ID to use')
    parser.add_argument('--exp_type', type=str, default='full-shot', help='[full-shot, zero-shot, few-shot]')
    # --- 模型结构参数 (Model Architecture) ---
    parser.add_argument('--resnet_base_filters', type=int, default=64, help='ResNet 基础通道数 (宽度)，越小越轻量 (e.g., 32, 64)')
    parser.add_argument('--resnet_depth', type=str, default='resnet18', choices=['resnet18', 'shallow'], help='ResNet 深度模式: "resnet18"=[2,2,2,2], "shallow"=[1,1,1,1]')
    parser.add_argument('--head_dropout', type=float, default=0.2, help='预测头部的 Dropout 比率 (防止过拟合)')
    parser.add_argument('--fusion_dropout', type=float, default=0.0, help='门控融合层的 Dropout 比率')

    
    args = parser.parse_args()

    # === 统一打印实验参数 ===
    print("\n===== Experiment Configuration =====")
    for k, v in sorted(vars(args).items()):
        print(f"{k:15}: {v}")
    print("===================================\n")


    if args.exp_type == 'full-shot':    
        finetune_exp = exp_down_fullshot.Exp_Finetune(args)
    
    elif args.exp_type == 'few-shot':
        finetune_exp = exp_down_fewshot.Exp_Finetune(args)    

    elif args.exp_type == 'zero-shot':
        finetune_exp = exp_down_zeroshot.Exp_Finetune(args)

    else:
        return

    finetune_exp = exp_down_fullshot.Exp_Finetune(args)

    finetune_exp.finetune()

main()