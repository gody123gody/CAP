import os
import numpy as np
import matplotlib.pyplot as plt

# ========== 配置：下游四个数据集窗口时长 ==========
DOWNSTREAM_INFO = {
    "fangchan": 25,   # s
    "xueya":   10,
    "xintiao": 30,
    "huxipinlu": 30,
}

DOWNSTREAM_ROOT = "/public/home/ai_user_1/DC/hcy/dataset/down_steam_dataset"

# 预训练 ED 路径（按你之前给的）
ED_NPZ_DIR = "/public/home/ai_user_1/DC/hcy/dataset/ed/outputs"

def analyze_signal(sig, fs):
    sig = sig.astype(np.float32)
    stats = {
        "len": int(sig.shape[0]),
        "min": float(sig.min()),
        "max": float(sig.max()),
        "mean": float(sig.mean()),
        "std": float(sig.std()),
        "p1": float(np.percentile(sig, 1)),
        "p99": float(np.percentile(sig, 99)),
    }

    x = sig - sig.mean()
    n = len(x)
    win = np.hanning(n).astype(np.float32)
    xw = x * win
    spec = np.fft.rfft(xw)
    mag = np.abs(spec) + 1e-12
    freqs = np.fft.rfftfreq(n, d=1.0/fs)

    # 去掉 DC
    if len(mag) > 1:
        mag[0] = 0.0
    peak_idx = int(np.argmax(mag))
    peak_hz = float(freqs[peak_idx])
    peak_bpm = float(peak_hz * 60.0)

    return stats, freqs, mag, peak_hz, peak_bpm

def plot_time_and_spec(sig, fs, title, max_sec=10):
    L = len(sig)
    seg_len = int(min(L, max_sec * fs))
    seg = sig[:seg_len]

    stats, freqs, mag, peak_hz, peak_bpm = analyze_signal(seg, fs)

    t = np.arange(seg_len) / fs
    plt.figure()
    plt.plot(t, seg)
    plt.xlabel("time (s)")
    plt.ylabel("ppg")
    plt.title(f"{title} | peak≈{peak_hz:.2f}Hz ({peak_bpm:.1f}bpm)")

    plt.figure()
    plt.plot(freqs, mag)
    plt.xlabel("frequency (Hz)")
    plt.ylabel("magnitude")
    plt.title(f"{title} spectrum")
    plt.show()

def to_ncl(X):
    X = np.asarray(X)
    if X.ndim == 2:
        return X[:, None, :]
    if X.ndim == 3:
        N, a, b = X.shape
        # (N,C,L)
        if a in (1,2,3,4) and b > 10:
            return X
        # (N,L,C)
        if b in (1,2,3,4) and a > 10:
            return np.transpose(X, (0,2,1))
    raise ValueError(f"Unexpected X shape: {X.shape}")

def summarize_peaks(X_ncl, fs, channel=0, sample_n=200, seed=0):
    rng = np.random.default_rng(seed)
    N, C, L = X_ncl.shape
    idxs = rng.choice(N, size=min(sample_n, N), replace=False)
    peaks = []
    for idx in idxs:
        sig = X_ncl[idx, channel].astype(np.float32)
        _, _, _, peak_hz, peak_bpm = analyze_signal(sig, fs)
        peaks.append(peak_bpm)
    peaks = np.array(peaks)
    return {
        "bpm_mean": float(peaks.mean()),
        "bpm_std": float(peaks.std()),
        "bpm_p5": float(np.percentile(peaks, 5)),
        "bpm_p50": float(np.percentile(peaks, 50)),
        "bpm_p95": float(np.percentile(peaks, 95)),
    }

def analyze_downstream_dataset(name, split="test", n_plot=3, channel=0):
    path = os.path.join(DOWNSTREAM_ROOT, name)
    X = np.load(os.path.join(path, "X_test.npy"))
    y = np.load(os.path.join(path, "y_test.npy"))
    X_ncl = to_ncl(X)

    N, C, L = X_ncl.shape
    sec = DOWNSTREAM_INFO[name]
    fs = L / sec

    # 全局统计
    flat = X_ncl.reshape(N*C, L)
    print("="*90)
    print(f"[DOWNSTREAM] {name} | window={sec}s | shape=(N,C,L)=({N},{C},{L}) | inferred fs={fs:.3f}Hz")
    print("global min/max:", float(X_ncl.min()), float(X_ncl.max()))
    print("mean/std over all points:", float(X_ncl.mean()), float(X_ncl.std()))
    print("mean p1/p99 (per-trace):",
          float(np.mean(np.percentile(flat, 1, axis=1))),
          float(np.mean(np.percentile(flat, 99, axis=1))))
    print("peak bpm summary:", summarize_peaks(X_ncl, fs, channel=channel, sample_n=200))

    # 随机画几条
    rng = np.random.default_rng(0)
    idxs = rng.choice(N, size=min(n_plot, N), replace=False)
    for i, idx in enumerate(idxs):
        sig = X_ncl[idx, channel].astype(np.float32)
        title = f"{name} test idx={idx} y={float(np.asarray(y[idx]).reshape(-1)[0]):.3f}"
        plot_time_and_spec(sig, fs, title, max_sec=min(10, sec))

def estimate_fs_from_timestamp(ts):
    ts = np.asarray(ts).astype(np.float64)
    if ts.ndim != 1 or ts.size < 2:
        return None
    dt = np.diff(ts)
    # 去掉非正/异常间隔
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        return None
    med = np.median(dt)
    if med <= 0:
        return None
    return 1.0 / med

def analyze_ed_samples(n_plot=3, seed=0):
    files = [f for f in os.listdir(ED_NPZ_DIR) if f.endswith(".npz")]
    if not files:
        print("[ED] no npz files found")
        return
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(files), size=min(n_plot, len(files)), replace=False)

    print("="*90)
    print("[ED] analyzing pretrain samples with timestamp-derived fs")

    for j, k in enumerate(picks):
        fn = files[int(k)]
        p = os.path.join(ED_NPZ_DIR, fn)
        arr = np.load(p)
        sig = arr["ppg_value"].astype(np.float32)
        ts = arr["timestamp"] if "timestamp" in arr.files else None

        if sig.ndim > 1:
            sig = sig[0]

        fs = None
        if ts is not None:
            fs = estimate_fs_from_timestamp(ts)

        print("-"*80)
        print(f"[ED] {fn} | len={len(sig)} | fs(from ts)={fs}")

        if fs is None:
            # 没法换算 Hz，就先按“点”为横轴画
            # 但你这份数据有 timestamp，正常应该能算出 fs
            fs = 1.0

        # 画前 10 秒
        plot_time_and_spec(sig, fs, f"ED {fn}", max_sec=10)

        # 输出统计 + 主峰 bpm
        stats, _, _, peak_hz, peak_bpm = analyze_signal(sig, fs)
        dur = len(sig) / fs
        print("duration(s):", float(dur))
        print("stats:", stats)
        print(f"peak≈{peak_hz:.3f}Hz ({peak_bpm:.1f} bpm)")

if __name__ == "__main__":
    # 1) 下游四个数据集逐个看
    for name in ["fangchan", "xueya", "xintiao", "huxipinlu"]:
        analyze_downstream_dataset(name, split="test", n_plot=2, channel=0)

    # 2) 预训练 ED 随机抽几条用 timestamp 反推 fs
    analyze_ed_samples(n_plot=2, seed=0)
