import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import wfdb
import os
import json
from torch.utils.data.dataloader import DataLoader
import pickle 

class train_MIMIC_Dataset(Dataset):
    def __init__(self,
                 ppg_dir='/public/home/ai_user_1/DC/hcy/dataset/mimic/ppg',
                 json_dir='/public/home/ai_user_1/DC/hcy/dataset/mimic/report',
                 cache_file='/public/home/ai_user_1/DC/hcy/PPG_Clip/mimic.pkl', # 1898
                 target_len: int | None = 37500,     # 设 None 则不在 Dataset 中 pad
                 do_zscore: bool = False):
        self.ppg_dir = ppg_dir
        self.json_dir = json_dir
        self.cache_file = cache_file
        self.target_len = target_len
        self.do_zscore = do_zscore

        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.sample_names = pickle.load(f)
            print(f"[INFO] Loaded cached sample names from {self.cache_file}, total={len(self.sample_names)}")
        else:
            sample_names1 = {os.path.splitext(f)[0] for f in os.listdir(self.ppg_dir) if f.endswith('.npz')}
            sample_names2 = {os.path.splitext(f)[0] for f in os.listdir(self.json_dir) if f.endswith('.json')}
            common_names = sorted(sample_names1 & sample_names2)

            valid_names = []
            for base in common_names:
                json_path = os.path.join(self.json_dir, base + ".json")
                try:
                    if os.path.getsize(json_path) == 0:  # 空文件
                        continue
                    with open(json_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if not content:  # 只有空白
                        continue
                    json.loads(content)  # 确认能解析
                    valid_names.append(base)
                except Exception:
                    # 解析失败，直接跳过
                    continue

            self.sample_names = valid_names
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.sample_names, f)

            print(f"[INFO] Cached {len(self.sample_names)} valid sample names to {self.cache_file} "
                f"(from {len(common_names)} intersected)")
    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        base = self.sample_names[idx]
        npz_path  = os.path.join(self.ppg_dir,  base + '.npz')
        json_path = os.path.join(self.json_dir, base + '.json')

        # 读取 PPG 信号
        arr = np.load(npz_path)
        # print(arr)
        # sig = arr["ppg_value"].astype(np.float32)   # [T] 或 [C, T]
        sig = arr["data"].astype(np.float32)   # [T] 或 [C, T]

        if sig.ndim > 1:   # 多通道只取第 1 通道
            sig = sig[0]

        length = len(sig)  # 原始长度
        # 统一到 target_len
        if length >= self.target_len:
            sig = sig[:self.target_len]
        else:
            pad = np.zeros(self.target_len - length, dtype=np.float32)
            sig = np.concatenate([sig, pad], axis=0)

        sig_t = torch.from_numpy(sig).float().unsqueeze(0)  # [1, target_len]
        
        sig_t = (sig_t - sig_t.mean(dim=-1, keepdim=True)) / (sig_t.std(dim=-1, keepdim=True) + 1e-8)

        # 读取文本
        # with open(json_path, 'r') as f:
        #     data = json.load(f)
        # diagnosis = data.get('Report', '')

        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                raise ValueError(f"Empty JSON file: {json_path}")
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in file {json_path}: {e}\nContent preview: {content[:200]}")
        diagnosis = data.get('Report', '')

        return {
            'ppg': sig_t,                               # [1, target_len]
            'txt': diagnosis
        }

class train_ED_Dataset(Dataset):
    def __init__(self,
                 ppg_dir='/public/home/ai_user_1/DC/hcy/dataset/ed/outputs',
                 json_dir='/public/home/ai_user_1/DC/hcy/dataset/ed/outputs_Llama',
                 cache_file='/public/home/ai_user_1/DC/hcy/PPG_Clip/ed.pkl',
                 target_len: int | None = 37500,     # 可选的最大长度
                 do_zscore: bool = False):
        self.ppg_dir = ppg_dir
        self.json_dir = json_dir
        self.cache_file = cache_file
        self.target_len = target_len
        self.do_zscore = do_zscore

        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.sample_names = pickle.load(f)
            print(f"[INFO] Loaded cached sample names from {self.cache_file}, total={len(self.sample_names)}")
        else:
            sample_names1 = {os.path.splitext(f)[0] for f in os.listdir(self.ppg_dir) if f.endswith('.npz')}
            sample_names2 = {os.path.splitext(f)[0] for f in os.listdir(self.json_dir) if f.endswith('.json')}
            common_names = sorted(sample_names1 & sample_names2)

            valid_names = []
            for base in common_names:
                json_path = os.path.join(self.json_dir, base + ".json")
                try:
                    if os.path.getsize(json_path) == 0:  # 空文件
                        continue
                    with open(json_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if not content:  # 只有空白
                        continue
                    json.loads(content)  # 确认能解析
                    valid_names.append(base)
                except Exception:
                    continue

            self.sample_names = valid_names
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.sample_names, f)

            print(f"[INFO] Cached {len(self.sample_names)} valid sample names to {self.cache_file} "
                f"(from {len(common_names)} intersected)")

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        base = self.sample_names[idx]
        npz_path  = os.path.join(self.ppg_dir,  base + '.npz')
        json_path = os.path.join(self.json_dir, base + '.json')

        # 读取 PPG 信号
        arr = np.load(npz_path)
        sig = arr["ppg_value"].astype(np.float32)   # [T] 或 [C, T]

        if sig.ndim > 1:   # 多通道只取第 1 通道
            sig = sig[0]

        if self.do_zscore:
            m = sig.mean()
            s = sig.std()
            if s < 1e-6:
                s = 1.0
            sig = (sig - m) / s

        length = len(sig)  # 原始长度
        # 返回原始长度的信号，不强制 pad
        sig_t = torch.from_numpy(sig).float().unsqueeze(0)  # [1, length]

        # 读取文本
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                raise ValueError(f"Empty JSON file: {json_path}")
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in file {json_path}: {e}\nContent preview: {content[:200]}")
        diagnosis = data.get('Report', '')

        return {
            'ppg': sig_t,                               # [1, length]
            'txt': diagnosis,
            'ppg_len': length                            # 返回信号长度，后续用于 padding 和 mask
        }


class val_Dataset(Dataset):
    def __init__(self):
        self.data_folder = '/public/home/ai_user_1/DC/hcy/dataset/PPG_val'
        # 筛选出所有 .csv 文件，并去除扩展名
        self.sample_names = [f[:-4] for f in os.listdir(self.data_folder) if f.endswith('.csv')]

    def __len__(self):
        return len(self.sample_names)

    def z_score_normalization(self, signal):
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    def __getitem__(self, idx):
        base_name = self.sample_names[idx]
        csv_path = os.path.join(self.data_folder, base_name + '.csv')
        json_path = os.path.join(self.data_folder, base_name + '.json')

        # 读取 PPG 信号
        df = pd.read_csv(csv_path)
        signal = df['ppg_value'].to_numpy(dtype=np.float32)
        signal = signal.squeeze(0)
        signal = self.z_score_normalization(signal)

        signal = torch.FloatTensor(signal)
        signal = torch.FloatTensor(signal).unsqueeze(0)  # 变成 shape [1, 37500]


        # 读取诊断文本
        with open(json_path, 'r') as f:
            data = json.load(f)
        diagnosis = data.get('Report', '')

        sample = {'ppg': signal, 'txt': diagnosis}
        return sample

