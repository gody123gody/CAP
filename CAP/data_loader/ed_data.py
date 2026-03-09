# import os
# import json
# import pickle
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import matplotlib.pyplot as plt

# # ====== 你的 dataset（按你给的代码几乎原样复制）======
# class train_ED_Dataset(Dataset):
#     def __init__(self,
#                  ppg_dir='/public/home/ai_user_1/DC/hcy/dataset/ed/outputs',
#                  json_dir='/public/home/ai_user_1/DC/hcy/dataset/ed/outputs_Llama',
#                  cache_file='/public/home/ai_user_1/DC/hcy/PPG_Clip/ed.pkl',
#                  target_len: int | None = 37500,
#                  do_zscore: bool = False):
#         self.ppg_dir = ppg_dir
#         self.json_dir = json_dir
#         self.cache_file = cache_file
#         self.target_len = target_len
#         self.do_zscore = do_zscore

#         if os.path.exists(self.cache_file):
#             with open(self.cache_file, "rb") as f:
#                 self.sample_names = pickle.load(f)
#             print(f"[INFO] Loaded cached sample names from {self.cache_file}, total={len(self.sample_names)}")
#         else:
#             sample_names1 = {os.path.splitext(f)[0] for f in os.listdir(self.ppg_dir) if f.endswith('.npz')}
#             sample_names2 = {os.path.splitext(f)[0] for f in os.listdir(self.json_dir) if f.endswith('.json')}
#             common_names = sorted(sample_names1 & sample_names2)

#             valid_names = []
#             for base in common_names:
#                 json_path = os.path.join(self.json_dir, base + ".json")
#                 try:
#                     if os.path.getsize(json_path) == 0:
#                         continue
#                     with open(json_path, "r", encoding="utf-8") as f:
#                         content = f.read().strip()
#                     if not content:
#                         continue
#                     json.loads(content)
#                     valid_names.append(base)
#                 except Exception:
#                     continue

#             self.sample_names = valid_names
#             with open(self.cache_file, "wb") as f:
#                 pickle.dump(self.sample_names, f)

#             print(f"[INFO] Cached {len(self.sample_names)} valid sample names to {self.cache_file} "
#                   f"(from {len(common_names)} intersected)")

#     def __len__(self):
#         return len(self.sample_names)

#     def __getitem__(self, idx):
#         base = self.sample_names[idx]
#         npz_path  = os.path.join(self.ppg_dir,  base + '.npz')
#         json_path = os.path.join(self.json_dir, base + '.json')

#         arr = np.load(npz_path)
#         sig = arr["ppg_value"].astype(np.float32)

#         if sig.ndim > 1:
#             sig = sig[0]

#         if self.do_zscore:
#             m = sig.mean()
#             s = sig.std()
#             if s < 1e-6:
#                 s = 1.0
#             sig = (sig - m) / s

#         length = len(sig)
#         sig_t = torch.from_numpy(sig).float().unsqueeze(0)

#         with open(json_path, 'r', encoding='utf-8') as f:
#             content = f.read().strip()
#             if not content:
#                 raise ValueError(f"Empty JSON file: {json_path}")
#             data = json.loads(content)

#         diagnosis = data.get('Report', '')

#         return {
#             'ppg': sig_t,
#             'txt': diagnosis,
#             'ppg_len': length,
#             'npz_path': npz_path,
#             'json_path': json_path
#         }


# def try_get_fs_from_npz(npz_obj):
#     # 常见采样率 key 猜测
#     cand_keys = ["fs", "FS", "sr", "SR", "sampling_rate", "SamplingRate", "sample_rate", "hz", "Hz"]
#     for k in cand_keys:
#         if k in npz_obj.files:
#             v = npz_obj[k]
#             try:
#                 v = float(np.array(v).reshape(-1)[0])
#                 if v > 0:
#                     return v, k
#             except:
#                 pass
#     return None, None


# def analyze_one(sig, fs=None, title_prefix="", max_plot_sec=10, max_plot_points=5000):
#     sig = np.asarray(sig).astype(np.float32)
#     T = sig.shape[0]

#     # basic stats
#     stats = {
#         "len": int(T),
#         "min": float(sig.min()),
#         "max": float(sig.max()),
#         "mean": float(sig.mean()),
#         "std": float(sig.std()),
#         "p1": float(np.percentile(sig, 1)),
#         "p99": float(np.percentile(sig, 99)),
#     }

#     # choose a segment to plot
#     if fs is not None:
#         seg_len = int(min(T, max_plot_sec * fs))
#     else:
#         seg_len = int(min(T, max_plot_points))
#     seg = sig[:seg_len]

#     # detrend for FFT (remove mean)
#     seg_d = seg - seg.mean()

#     # FFT
#     n = len(seg_d)
#     win = np.hanning(n).astype(np.float32)
#     seg_w = seg_d * win
#     spec = np.fft.rfft(seg_w)
#     mag = np.abs(spec) + 1e-12
#     freqs = np.fft.rfftfreq(n, d=(1.0/fs) if fs is not None else 1.0)

#     # ignore DC, find peak
#     mag2 = mag.copy()
#     if len(mag2) > 1:
#         mag2[0] = 0.0
#     peak_idx = int(np.argmax(mag2))
#     peak_f = float(freqs[peak_idx])

#     # if fs known, convert to bpm
#     peak_bpm = None
#     if fs is not None:
#         peak_bpm = peak_f * 60.0

#     # plot
#     plt.figure()
#     if fs is not None:
#         t = np.arange(seg_len) / fs
#         plt.plot(t, seg)
#         plt.xlabel("time (s)")
#     else:
#         plt.plot(seg)
#         plt.xlabel("sample index")
#     plt.ylabel("ppg value")
#     plt.title(f"{title_prefix} time-domain (len={T})")

#     plt.figure()
#     if fs is not None:
#         plt.plot(freqs, mag)
#         plt.xlabel("frequency (Hz)")
#         plt.title(f"{title_prefix} spectrum | peak={peak_f:.3f} Hz ({peak_bpm:.1f} bpm)" if peak_bpm is not None else
#                   f"{title_prefix} spectrum | peak={peak_f:.3f}")
#     else:
#         plt.plot(freqs, mag)
#         plt.xlabel("frequency (cycles/sample)")
#         plt.title(f"{title_prefix} spectrum | peak={peak_f:.6f} cycles/sample")

#     plt.ylabel("magnitude")

#     plt.show()

#     return stats, peak_f, peak_bpm


# def main(
#     n_samples=5,
#     fs_default=None,   # 如果 npz 没有采样率字段，你在这填一个，比如 125/250/300/500
#     seed=0
# ):
#     ds = train_ED_Dataset(do_zscore=False)

#     rng = np.random.default_rng(seed)
#     idxs = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

#     for j, idx in enumerate(idxs):
#         item = ds[int(idx)]
#         npz_path = item["npz_path"]
#         arr = np.load(npz_path)

#         fs, fs_key = try_get_fs_from_npz(arr)
#         if fs is None:
#             fs = fs_default

#         sig = item["ppg"].squeeze(0).numpy()

#         print("=" * 80)
#         print(f"[{j+1}/{len(idxs)}] idx={idx}")
#         print("npz:", npz_path)
#         print("npz keys:", arr.files)
#         print("ppg_len:", item["ppg_len"])
#         print("fs:", fs, f"(from {fs_key})" if fs_key is not None else "(default/unknown)")

#         stats, peak_f, peak_bpm = analyze_one(
#             sig,
#             fs=fs,
#             title_prefix=f"sample {j+1}",
#             max_plot_sec=10,
#             max_plot_points=5000
#         )

#         print("stats:", stats)
#         if fs is not None:
#             print(f"peak frequency: {peak_f:.3f} Hz, approx: {peak_bpm:.1f} bpm")
#         else:
#             print(f"peak frequency: {peak_f:.6f} cycles/sample (need fs to convert to Hz/bpm)")


# if __name__ == "__main__":
#     # 如果你不知道采样率，先设 None，只看相对频谱
#     # 如果你大概知道，比如 125Hz，就填 125
#     main(n_samples=5, fs_default=None, seed=0)


import os
import numpy as np
from collections import Counter

ppg_dir = "/public/home/ai_user_1/DC/hcy/dataset/mimic/ppg"  # 改这里
key_name = "data"  # 你现在用的是 arr["data"]

lengths = []
bad = []

files = [f for f in os.listdir(ppg_dir) if f.endswith(".npz")]
files.sort()

for f in files:
    p = os.path.join(ppg_dir, f)
    try:
        arr = np.load(p)
        # print(p.shape)
        if key_name not in arr.files:
            bad.append((f, f"missing key {key_name}, keys={arr.files}"))
            continue
        sig = arr[key_name]
        print(sig.shape)
        # 多通道只取第1通道
        if sig.ndim > 1:
            sig = sig[0]
        lengths.append(int(sig.shape[-1]))
    except Exception as e:
        bad.append((f, repr(e)))

lengths = np.array(lengths, dtype=np.int64)

print("ppg_dir:", ppg_dir)
print("total npz:", len(files))
print("valid:", len(lengths), "bad:", len(bad))

if len(bad) > 0:
    print("\nExamples of bad files (up to 10):")
    for x in bad[:10]:
        print("  ", x)

if len(lengths) == 0:
    raise SystemExit("No valid signals found.")

# 基本统计
print("\nLength stats:")
print("min:", int(lengths.min()))
print("max:", int(lengths.max()))
print("mean:", float(lengths.mean()))
print("std:", float(lengths.std()))
for q in [1, 5, 50, 95, 99]:
    print(f"p{q}:", int(np.percentile(lengths, q)))

# 关心 37500
target = 37500
eq = int((lengths == target).sum())
lt = int((lengths < target).sum())
gt = int((lengths > target).sum())
print("\nCompare to target_len=37500:")
print("==37500:", eq)
print("<37500:", lt)
print(">37500:", gt)

# 最常见长度
cnt = Counter(lengths.tolist())
most_common = cnt.most_common(15)
print("\nMost common lengths (top 15):")
for L, c in most_common:
    print(f"len={L}: {c}")

# 如果你想看看有哪些“不是37500”的长度样例
non_target = lengths[lengths != target]
print("\nnon-37500 count:", int(non_target.size))
if non_target.size > 0:
    uniq = np.unique(non_target)
    print("unique non-37500 lengths (up to 30):", uniq[:30].tolist())
