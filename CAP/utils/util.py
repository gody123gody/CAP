import math
import os
import shutil
import numpy as np
from time import gmtime, strftime
from matplotlib import pyplot as plt
from collections import Counter, OrderedDict


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix, balanced_accuracy_score, roc_curve
from sklearn.utils import resample
from sklearn.metrics import average_precision_score,auc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'")

from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def my_eval_with_ci_thresh(
    gt, 
    pred, 
    n_bootstrap=1000, 
    ci_percentile=95
):
    """
    Evaluates the model with a fixed threshold of 0.5 for each task, 
    and computes AUROC, AUPRC, PPV, NPV, sensitivity, specificity
    (all with confidence intervals).

    Args:
        gt: Ground truth labels (numpy array), shape: [N, n_task]
        pred: Prediction probabilities (numpy array), shape: [N, n_task]
        n_bootstrap: Number of bootstrap samples to generate for confidence intervals
        ci_percentile: Percentile for the confidence intervals (e.g., 95 for 95% CI)

    Returns:
        mean_metrics_dict: dict, 键是各指标的名称，值是在所有task上平均后的数值 (float)
        metrics_per_task_dict: dict, 键是各指标的名称，值是一个 ndarray (长度 n_task)
        ci_per_task_dict: dict, 键是各指标的名称，值是对应指标在每个 task 的 (lower, upper) CI 列表
    """
    n_task = gt.shape[1]
    
    # 各指标在每个 task 上的取值
    rocaucs = []
    auprcs  = []
    ppvs    = []
    npvs    = []
    sensitivities = []
    specificities = []
    
    # 用于存储“每个 task 的置信区间 (lower, upper)”
    rocauc_cis = []
    auprc_cis  = []
    ppv_cis    = []
    npv_cis    = []
    sens_cis   = []
    spec_cis   = []

    for i in range(n_task):
        tmp_gt = np.nan_to_num(gt[:, i], nan=0)
        tmp_pred = np.nan_to_num(pred[:, i], nan=0)

        # ============== 1) 计算 AUROC
        try:
            auroc_val = roc_auc_score(tmp_gt, tmp_pred)
        except ValueError:
            auroc_val = 0.0
        rocaucs.append(auroc_val)

        # ============== 2) 计算 AUPRC
        try:
            auprc_val = average_precision_score(tmp_gt, tmp_pred)
        except ValueError:
            auprc_val = 0.0
        auprcs.append(auprc_val)

        # ============== 根据阈值 0.5 获取预测标签
        pred_labels = (tmp_pred > 0.5).astype(int)
        cm = confusion_matrix(tmp_gt, pred_labels).ravel()

        # confusion_matrix 返回值可能长度不一，需要做兼容
        if len(cm) == 1:
            # 只有正类或只有负类
            if pred_labels.sum() == 0:  # Only negative class predicted
                tn, fp, fn, tp = cm[0], 0, 0, 0
            else:  # Only positive class predicted
                tn, fp, fn, tp = 0, 0, 0, cm[0]
        else:
            tn, fp, fn, tp = cm

        # ============== 3) sensitivity (recall)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivities.append(sensitivity)

        # ============== 4) specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

        # ============== 5) PPV (precision)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        ppvs.append(ppv)

        # ============== 6) NPV
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        npvs.append(npv)

        # ============== 对每个 task 做 bootstrap CI
        rocauc_ci = bootstrap_ci(
            gt[:, i], pred[:, i], metric='roc_auc',
            n_bootstrap=n_bootstrap,
            ci_percentile=ci_percentile
        )
        auprc_ci = bootstrap_ci(
            gt[:, i], pred[:, i], metric='auprc',
            n_bootstrap=n_bootstrap,
            ci_percentile=ci_percentile
        )
        ppv_ci = bootstrap_ci(
            gt[:, i], pred[:, i], metric='ppv',
            n_bootstrap=n_bootstrap,
            ci_percentile=ci_percentile
        )
        npv_ci = bootstrap_ci(
            gt[:, i], pred[:, i], metric='npv',
            n_bootstrap=n_bootstrap,
            ci_percentile=ci_percentile
        )
        sens_ci = bootstrap_ci(
            gt[:, i], pred[:, i], metric='sensitivity',
            n_bootstrap=n_bootstrap,
            ci_percentile=ci_percentile
        )
        spec_ci = bootstrap_ci(
            gt[:, i], pred[:, i], metric='specificity',
            n_bootstrap=n_bootstrap,
            ci_percentile=ci_percentile
        )

        rocauc_cis.append(rocauc_ci)
        auprc_cis.append(auprc_ci)
        ppv_cis.append(ppv_ci)
        npv_cis.append(npv_ci)
        sens_cis.append(sens_ci)
        spec_cis.append(spec_ci)

    # 转成 numpy array 方便后续求 mean
    rocaucs = np.array(rocaucs)
    auprcs  = np.array(auprcs)
    ppvs    = np.array(ppvs)
    npvs    = np.array(npvs)
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)

    # 计算每个指标在所有 task 上的平均值
    mean_rocauc = np.mean(rocaucs)
    mean_auprc  = np.mean(auprcs)
    mean_ppv    = np.mean(ppvs)
    mean_npv    = np.mean(npvs)
    mean_sens   = np.mean(sensitivities)
    mean_spec   = np.mean(specificities)

    # 将结果打包返回
    mean_metrics_dict = {
        "AUROC": mean_rocauc,
        "AUPRC": mean_auprc,
        "PPV": mean_ppv,
        "NPV": mean_npv,
        "Sensitivity": mean_sens,
        "Specificity": mean_spec
    }
    metrics_per_task_dict = {
        "AUROC": rocaucs,
        "AUPRC": auprcs,
        "PPV": ppvs,
        "NPV": npvs,
        "Sensitivity": sensitivities,
        "Specificity": specificities
    }
    ci_per_task_dict = {
        "AUROC": rocauc_cis,
        "AUPRC": auprc_cis,
        "PPV": ppv_cis,
        "NPV": npv_cis,
        "Sensitivity": sens_cis,
        "Specificity": spec_cis
    }

    return mean_metrics_dict, metrics_per_task_dict, ci_per_task_dict

def bootstrap_ci(
    gt, 
    pred, 
    metric, 
    n_bootstrap=1000, 
    ci_percentile=95
):
    """
    Calculates confidence intervals for a given metric using bootstrapping.

    Args:
        gt: Ground truth labels (numpy array), shape: [N,]
        pred: Prediction probabilities (numpy array), shape: [N,]
        metric: One of ['roc_auc', 'auprc', 'ppv', 'npv', 'sensitivity', 'specificity']
        n_bootstrap: Number of bootstrap samples to generate
        ci_percentile: Percentile for the confidence intervals

    Returns:
        (lower_bound, upper_bound): tuple of floats
    """
    from sklearn.metrics import (roc_auc_score, average_precision_score, 
                                 confusion_matrix, f1_score)

    n = len(gt)
    # 用于存储每次 bootstrap 后的度量值
    metrics_list = []

    for _ in range(n_bootstrap):
        # 随机有放回抽样
        indices = np.random.choice(range(n), size=n, replace=True)
        gt_resampled = gt[indices]
        pred_resampled = pred[indices]

        # 计算指标
        if metric == 'roc_auc':
            try:
                val = roc_auc_score(gt_resampled, pred_resampled)
            except ValueError:
                val = 0.0

        elif metric == 'auprc':
            try:
                val = average_precision_score(gt_resampled, pred_resampled)
            except ValueError:
                val = 0.0

        else:
            # 先基于 0.5 获取预测标签
            pred_labels = (pred_resampled > 0.5).astype(int)
            cm = confusion_matrix(gt_resampled, pred_labels).ravel()
            if len(cm) == 1:
                if pred_labels.sum() == 0:  
                    tn, fp, fn, tp = cm[0], 0, 0, 0
                else:
                    tn, fp, fn, tp = 0, 0, 0, cm[0]
            else:
                tn, fp, fn, tp = cm

            if metric == 'sensitivity':   # recall
                val = tp / (tp + fn) if (tp + fn) > 0 else 0
            elif metric == 'specificity':
                val = tn / (tn + fp) if (tn + fp) > 0 else 0
            elif metric == 'ppv':  # precision
                val = tp / (tp + fp) if (tp + fp) > 0 else 0
            elif metric == 'npv':
                val = tn / (tn + fn) if (tn + fn) > 0 else 0
            else:
                # 如果有其他分支，比如 f1_score，可以在这里继续 elif
                val = 0.0

        metrics_list.append(val)

    # 对多次 bootstrap 的结果取 percentile，得到区间下、上界
    alpha = (100 - ci_percentile) / 2
    lower_bound = np.percentile(metrics_list, alpha)
    upper_bound = np.percentile(metrics_list, 100 - alpha)
    return (lower_bound, upper_bound)

def quantile_accuracy(y_true, y_pred, quantiles):
    """
    :param y_true: 
    :param y_pred: 
    :param quantiles: e.g. [0.25, 0.5, 0.75]
    """
    quantile_errors = {}
    for q in quantiles:
        pred_quantile = np.percentile(y_pred, q * 100)
        true_quantile = np.percentile(y_true, q * 100)
        # calculate error
        quantile_errors[q] = abs(pred_quantile - true_quantile)
    
    return quantile_errors

def find_optimal_thresholds(gt, pred):
    """
    Find optimal threshold for each task based on Balanced Accuracy.

    Args:
        gt: Ground truth labels (numpy array)
        pred: Prediction probabilities (numpy array)

    Returns:
        optimal_thresholds: Optimal threshold for each task
    """
    n_task = gt.shape[1]
    optimal_thresholds = []

    for i in range(n_task):
        best_ba = -1  
        best_thresh = 0.5  
        for thresh in np.linspace(0.01, 0.99, 99):  
            pred_labels = (pred[:, i] > thresh).astype(int)
            ba = balanced_accuracy_score(gt[:, i], pred_labels)  
            if ba > best_ba:
                best_ba = ba
                best_thresh = thresh
        optimal_thresholds.append(best_thresh)

    return optimal_thresholds


def compute_auc_with_ci(gt_np, pred_np, num_bootstrap=10, alpha=0.05):
    """Calculate AUC and its confidence interval using bootstrapping."""
    fpr, tpr, _ = roc_curve(gt_np, pred_np)
    roc_auc = auc(fpr, tpr)
    
    # Bootstrapping
    rng = np.random.default_rng()
    aucs = []
    for _ in range(num_bootstrap):
        indices = rng.choice(len(gt_np), len(gt_np), replace=True)
        if len(np.unique(gt_np[indices])) < 2:  # Ensure at least one positive and one negative
            continue
        fpr_boot, tpr_boot, _ = roc_curve(gt_np[indices], pred_np[indices])
        aucs.append(auc(fpr_boot, tpr_boot))
    
    # Calculate confidence intervals
    lower = np.percentile(aucs, 100 * alpha / 2)
    upper = np.percentile(aucs, 100 * (1 - alpha / 2))
    return roc_auc, (lower, upper)


def eval_with_dynamic_thresh_and_roc(gt, pred, task, run_id, lead,model_type):
    """
    Evaluate model and draw ROC curves separately for each task using task names.

    Args:
        gt: Ground truth labels (numpy array) [n_samples, n_tasks]
        pred: Prediction scores (numpy array) [n_samples, n_tasks]
        task: List of task names (length = n_tasks)

    Returns:
        mean_rocauc, rocaucs, sensitivities, specificities, f1s
    """
    save_dir = f'/data1/1shared/hcy/ECG-EchoReport/visual_AUC/{run_id}/{lead}_lead_FT_{model_type}'
    optimal_thresholds = find_optimal_thresholds(gt, pred)
    n_task = gt.shape[1]

    rocaucs = []
    sensitivities = []
    specificities = []
    f1s = []

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(n_task):
        tmp_gt = np.nan_to_num(gt[:, i], nan=0)
        tmp_pred = np.nan_to_num(pred[:, i], nan=0)

        # ROC-AUC
        try:
            # roc_auc = roc_auc_score(tmp_gt, tmp_pred)
            roc_auc, (ci_lower, ci_upper) = compute_auc_with_ci(tmp_gt, tmp_pred)
        except:
            roc_auc = 0.0
        rocaucs.append(roc_auc)

        # ROC curve
        fpr, tpr, thresholds = roc_curve(tmp_gt, tmp_pred)

        # Find optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]

        # Predict labels based on optimal threshold
        pred_labels = (tmp_pred > optimal_thresholds[i]).astype(int)

        # Confusion matrix
        cm = confusion_matrix(tmp_gt, pred_labels).ravel()
        if len(cm) == 1:
            if pred_labels.sum() == 0:
                tn, fp, fn, tp = cm[0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0]
        else:
            tn, fp, fn, tp = cm

        # Sensitivity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivities.append(sensitivity)

        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

        # F1 score
        f1 = f1_score(tmp_gt, pred_labels)
        f1s.append(f1)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}])')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.scatter(optimal_fpr, optimal_tpr, marker='o', color='red', label='Optimal Point', s=80)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {task[i]}')
        plt.legend(loc='lower right')
        plt.grid(False)

        # Save each figure with disease name
        if save_dir:
            # 防止中文名出现非法字符或者空格影响文件保存
            filename = task[i].replace('/', '_').replace(' ', '_')
            plt.savefig(os.path.join(save_dir, f'{i}_{filename}.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    # Convert lists to numpy arrays
    rocaucs = np.array(rocaucs)
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    f1s = np.array(f1s)

    # Mean ROC-AUC across tasks
    mean_rocauc = np.mean(rocaucs)

    return mean_rocauc, rocaucs, sensitivities, specificities, f1s

def my_eval_with_dynamic_thresh(gt, pred):
    """
    Evaluates the model with dynamically adjusted thresholds for each task.

    Args:
        gt: Ground truth labels (numpy array)
        pred: Prediction probabilities (numpy array)

    Returns:
        - Overall mean of the metrics across all tasks
        - Per-metric mean across all tasks (as a list)
        - All metrics per task in a columnar format
    """
    optimal_thresholds = find_optimal_thresholds(gt, pred)
    n_task = gt.shape[1]
    rocaucs = []
    sensitivities = []
    specificities = []
    f1 = []
    auprcs = []  # Step 2: Initialize list for AUPRC

    for i in range(n_task):
        tmp_gt = np.nan_to_num(gt[:, i], nan=0)
        tmp_pred = np.nan_to_num(pred[:, i], nan=0)

        # ROC-AUC
        try:
            rocaucs.append(roc_auc_score(tmp_gt, tmp_pred))
        except:
            rocaucs.append(0.0)

        # AUPRC  # Step 3: Calculate AUPRC
        try:
            auprc = average_precision_score(tmp_gt, tmp_pred)
            auprcs.append(auprc)
        except:
            auprcs.append(0.0)

        # Sensitivity and Specificity
        pred_labels = (tmp_pred > optimal_thresholds[i]).astype(int)
        # pred_labels = (tmp_pred > 0.5).astype(int)
        cm = confusion_matrix(tmp_gt, pred_labels).ravel()
        
        # Handle different sizes of confusion matrix
        if len(cm) == 1:
            # Only one class present in predictions
            if pred_labels.sum() == 0:  # Only negative class predicted
                tn, fp, fn, tp = cm[0], 0, 0, 0
            else:                       # Only positive class predicted
                tn, fp, fn, tp = 0, 0, 0, cm[0]
        else:
            tn, fp, fn, tp = cm

        # Calculate Sensitivity (True Positive Rate)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivities.append(sensitivity)
        
        # Calculate Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

        f1s = f1_score(tmp_gt, pred_labels)
        f1.append(f1s)
    
    # Convert lists to numpy arrays
    rocaucs = np.array(rocaucs)
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    f1 = np.array(f1)
    auprcs = np.array(auprcs)  # Step 4: Compute mean AUPRC

    # Calculate means for each metric
    mean_rocauc = np.mean(rocaucs)
    mean_auprc = np.mean(auprcs)  # Step 4: Compute mean AUPRC

    # Step 5: Update return statement
    return mean_rocauc, rocaucs, sensitivities, specificities, f1, auprcs, optimal_thresholds


def my_eval_new(gt, pred):
    thresh = 0.5
    n_task = gt.shape[1]
    res = []

    for i in range(n_task):
        tmp_res = []
        tmp_gt = gt[:, i]
        tmp_pred = pred[:, i]
        
        # 检查NaN并替换
        tmp_gt = np.nan_to_num(tmp_gt, nan=0)  # 可以选择适合你情况的替换值
        tmp_pred = np.nan_to_num(tmp_pred, nan=0)  # 同上
        
        tmp_pred_binary = np.array(tmp_pred > thresh, dtype=float)

        try:
            tmp_res.append(roc_auc_score(tmp_gt, tmp_pred))
        except Exception as e:
            tmp_res.append(-1.0)
        

        res.append(tmp_res)
    
    res = np.array(res)
    return np.mean(res, axis=0), res[:,0]

def my_eval(gt, pred):
    """
    gt, pred are from multi-task
    
    Returns:
    - Overall mean of the metrics across all tasks
    - Per-metric mean across all tasks (as a list)
    - All metrics per task in a columnar format
    """
    thresh = 0.5

    n_task = gt.shape[1]
    # Initialize lists for each metric
    rocaucs = []
    sensitivities = []
    specificities = []
    for i in range(n_task):
        tmp_gt = gt[:, i]
        tmp_pred = pred[:, i]
        tmp_gt = np.nan_to_num(tmp_gt, nan=0)  # 可以选择适合你情况的替换值
        tmp_pred = np.nan_to_num(tmp_pred, nan=0)  # 同上
        # ROC-AUC
        try:
            rocaucs.append(roc_auc_score(tmp_gt, tmp_pred))
        except:
            rocaucs.append(0.0)

        # Sensitivity and Specificity
        pred_labels = (tmp_pred > thresh).astype(int)
        cm = confusion_matrix(tmp_gt, pred_labels).ravel()
        
        # Handle different sizes of confusion matrix
        if len(cm) == 1:
            # Only one class present in predictions
            if pred_labels.sum() == 0:  # Only negative class predicted
                tn, fp, fn, tp = cm[0], 0, 0, 0
            else:                       # Only positive class predicted
                tn, fp, fn, tp = 0, 0, 0, cm[0]
        else:
            tn, fp, fn, tp = cm
        # Calculate Sensitivity (True Positive Rate)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivities.append(sensitivity)
        
        # Calculate Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

    # Convert lists to numpy arrays for easier mean calculation and handling
    rocaucs = np.array(rocaucs)
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)

    # Calculate means for each metric
    mean_rocauc = np.mean(rocaucs)

    return mean_rocauc, rocaucs, sensitivities, specificities

# def bootstrap_ci(gt, pred, n_bootstrap=1000, ci=0.95):
#     """
#     计算AUC的置信区间 (CI)。
    
#     参数：
#     gt: Ground truth (真实标签), 数组或列表
#     pred: Predictions (预测分数), 数组或列表
#     n_bootstrap: 引导法的迭代次数
#     ci: 置信水平，默认0.95
    
#     返回：
#     置信区间的下限和上限
#     """
#     bootstrapped_scores = []
#     n_size = len(gt)
    
#     # 引导法采样并计算AUC
#     for i in range(n_bootstrap):
#         # 有放回抽样同时对gt和pred进行抽样
#         sample_gt, sample_pred = resample(gt, pred, n_samples=n_size)
#         try:
#             # 计算AUC并保存结果
#             score = roc_auc_score(sample_gt, sample_pred)
#             bootstrapped_scores.append(score)
#         except ValueError:
#             # 如果出现无法计算AUC的情况（如正负类不足），忽略该次迭代
#             continue
    
#     # 计算置信区间
#     sorted_scores = np.sort(bootstrapped_scores)
#     lower_bound = np.percentile(sorted_scores, ((1 - ci) / 2) * 100)
#     upper_bound = np.percentile(sorted_scores, (ci + (1 - ci) / 2) * 100)
    
#     return lower_bound, upper_bound

def my_eval_new_with_ci(gt, pred, n_bootstrap=10, ci=0.95):
    thresh = 0.5
    n_task = gt.shape[1]
    res = []
    ci_res = []

    for i in range(n_task):
        tmp_res = []
        tmp_gt = gt[:, i]
        tmp_pred = pred[:, i]
        
        # 检查NaN并替换
        tmp_gt = np.nan_to_num(tmp_gt, nan=0)  # 可以选择适合你情况的替换值
        tmp_pred = np.nan_to_num(tmp_pred, nan=0)  # 同上
        
        tmp_pred_binary = np.array(tmp_pred > thresh, dtype=float)

        try:
            auc_score = roc_auc_score(tmp_gt, tmp_pred)
            tmp_res.append(auc_score)
        except Exception as e:
            tmp_res.append(0.0)
        
        # 引导法计算AUC的置信区间
        if tmp_res[0] != 0.0:
            # 引导法使用原始预测结果与gt
            lower_ci, upper_ci = bootstrap_ci(tmp_gt, tmp_pred, n_bootstrap=n_bootstrap, ci=ci)
            ci_res.append([lower_ci, upper_ci])
        else:
            ci_res.append([0.0, 0.0])

        res.append(tmp_res)
    
    res = np.array(res)
    ci_res = np.array(ci_res)
    return np.mean(res, axis=0), res[:, 0], ci_res

def get_time_str():
    return strftime("%Y%m%d_%H%M%S", gmtime())

def print_and_log(log_name, my_str):
    out = '{}|{}'.format(get_time_str(), my_str)
    print(out)
    with open(log_name, 'a') as f_log:
        print(out, file=f_log)

def save_checkpoint(state, path):
    filename = 'checkpoint_{0}_{1:.4f}.pth'.format(state['step'], state['val_rmse'])
    filename = os.path.join(path, filename)
    torch.save(state, filename)

def save_reg_checkpoint(state, path):
    filename = 'checkpoint_{0}_{1:.4f}.pth'.format(state['step'], state['mae'])
    filename = os.path.join(path, filename)
    torch.save(state, filename)