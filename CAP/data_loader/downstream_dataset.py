import os
import numpy as np
import re
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

# def load_dataloaders(dataset_path, batch_size=32, shuffle=True, num_workers=0):
#     """
#     加载指定路径下的数据集为 PyTorch DataLoader
    
#     返回：
#         train_loaders: List of DataLoader for each CV fold (训练集)
#         val_loaders:   List of DataLoader for each CV fold (验证集)
#         test_loader:   Single DataLoader for测试集
#     """
#     # 获取所有文件名
#     files = os.listdir(dataset_path)

#     # 读取测试集
#     X_test = np.load(os.path.join(dataset_path, 'X_test.npy'))
#     y_test = np.load(os.path.join(dataset_path, 'y_test.npy'))

#     # 转换为TensorDataset
#     test_dataset = TensorDataset(torch.from_numpy(X_test).float(),
#                                  torch.from_numpy(y_test).float())
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     # 自动检测 CV 数量
#     cv_pattern = re.compile(r'_CV(\d+)\.npy$')
#     cv_indices = set()
#     for f in files:
#         match = cv_pattern.search(f)
#         if match:
#             cv_indices.add(int(match.group(1)))
#     cv_indices = sorted(list(cv_indices))

#     train_loaders = []
#     val_loaders = []

#     for cv in cv_indices:
#         # 加载 train/val 数据
#         X_train = np.load(os.path.join(dataset_path, f'X_train_CV{cv}.npy'))
#         y_train = np.load(os.path.join(dataset_path, f'y_train_CV{cv}.npy'))
#         X_val = np.load(os.path.join(dataset_path, f'X_val_CV{cv}.npy'))
#         y_val = np.load(os.path.join(dataset_path, f'y_val_CV{cv}.npy'))

#         # 构造 TensorDataset
#         train_dataset = TensorDataset(torch.from_numpy(X_train).float(),
#                                       torch.from_numpy(y_train).float())
#         val_dataset = TensorDataset(torch.from_numpy(X_val).float(),
#                                     torch.from_numpy(y_val).float())

#         # 构造 DataLoader
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#         train_loaders.append(train_loader)
#         val_loaders.append(val_loader)

#     return train_loaders, val_loaders, test_loader



def load_dataloaders(dataset_path, batch_size=32, shuffle=True, num_workers=0):
    """
    加载指定路径下的数据集为 PyTorch DataLoader
    
    返回：
        cv_loaders: List[Dict]，每个元素包含一个fold的{'train': ..., 'val': ...} dataloader
        test_loader: 统一的测试集 dataloader
    """
    # 获取所有文件名
    files = os.listdir(dataset_path)

    # 加载测试集
    X_test = np.load(os.path.join(dataset_path, 'X_test.npy'))
    y_test = np.load(os.path.join(dataset_path, 'y_test.npy'))
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(),
                                 torch.from_numpy(y_test).float())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 自动检测 CV 编号
    cv_pattern = re.compile(r'_CV(\d+)\.npy$')
    cv_indices = set()
    for f in files:
        match = cv_pattern.search(f)
        if match:
            cv_indices.add(int(match.group(1)))
    cv_indices = sorted(list(cv_indices))

    # 构造 CV 加载器组合
    cv_loaders = []

    for cv in cv_indices:
        # 加载数据
        X_train = np.load(os.path.join(dataset_path, f'X_train_CV{cv}.npy'))
        y_train = np.load(os.path.join(dataset_path, f'y_train_CV{cv}.npy'))
        X_val = np.load(os.path.join(dataset_path, f'X_val_CV{cv}.npy'))
        y_val = np.load(os.path.join(dataset_path, f'y_val_CV{cv}.npy'))

        train_dataset = TensorDataset(torch.from_numpy(X_train).float(),
                                      torch.from_numpy(y_train).float())
        val_dataset = TensorDataset(torch.from_numpy(X_val).float(),
                                    torch.from_numpy(y_val).float())

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # 组合为一个fold的字典
        cv_loaders.append({
            'train': train_loader,
            'val': val_loader
        })

    return cv_loaders, test_loader


# cv_loaders, test_loader = load_dataloaders('/gemini/platform/public/aigc/Lirui/chengding/val_dataset/fangchan', batch_size=64)

# for i, fold in enumerate(cv_loaders):
#     print(f"Fold {i+1}: Train batches: {len(fold['train'])}, Val batches: {len(fold['val'])}")



# dataset_path = '/gemini/platform/public/aigc/Lirui/chengding/val_dataset/fangchan'
# train_loaders, val_loaders, test_loader = load_dataloaders(dataset_path, batch_size=64)

# for i, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
#     print(f"Fold {i+1}:")
#     print(f"  Train batches: {len(train_loader)}")
#     print(f"  Val batches: {len(val_loader)}")

# print(f"Test batches: {len(test_loader)}")