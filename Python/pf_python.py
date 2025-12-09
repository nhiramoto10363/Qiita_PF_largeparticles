"""
粒子フィルタ Python実装（Numba SIMD + Parallel 最適化版）

【大規模データ検証用設定】
- parallel=True: 有効化。N > 10^5 で効果を発揮します。
- num_particles: 1,000,000 (100万粒子) をデフォルトとします。
"""

import time
import numpy as np
import pandas as pd
from numba import njit, prange, get_num_threads
from math import lgamma, pi, log, exp, cos
from tqdm import tqdm

# コンパイルオプション
# parallel=True: マルチスレッド並列化を有効にする
# 並列計算用（予測、尤度、コピーなど）
JIT_PARALLEL = {'fastmath': True, 'parallel': True, 'cache': True}

# 直列計算用（系統的リサンプリングのインデックス生成など）
JIT_SERIAL   = {'fastmath': True, 'parallel': False, 'cache': True}


# ==========================================
# 修正箇所: 各関数のデコレータを使い分ける
# ==========================================

# ---- 予測ステップ（変更なし、PARALLELを使用）----
@njit(**JIT_PARALLEL)
def predict_linear_inplace(particles, noise):
    particles += noise

@njit(**JIT_PARALLEL)
def predict_gordon_inplace(particles, noise, t):
    forcing = 8.0 * cos(1.2 * t)
    particles[:] = (
        0.5 * particles
        + 25.0 * particles / (1.0 + particles * particles)
        + forcing
        + noise
    )

@njit(**JIT_PARALLEL)
def predict_positive_rw_inplace(particles, noise, eps):
    particles += noise
    particles[:] = np.maximum(particles, eps)


# ---- 尤度計算（変更なし、PARALLELを使用）----
@njit(**JIT_PARALLEL)
def loglik_gaussian_inplace(y_t, particles, sigma, out_loglik):
    const = -0.5 * log(2.0 * pi * sigma * sigma)
    inv_var = 1.0 / (sigma * sigma)
    out_loglik[:] = const - 0.5 * (y_t - particles)**2 * inv_var

@njit(**JIT_PARALLEL)
def loglik_poisson_inplace(y_t, particles, out_loglik):
    log_gamma_y = lgamma(y_t + 1.0)
    out_loglik[:] = particles
    out_loglik[:] = np.maximum(out_loglik, 1e-8)
    out_loglik[:] = y_t * np.log(out_loglik) - out_loglik - log_gamma_y


# ---- 重み正規化（変更なし、PARALLELを使用）----
@njit(**JIT_PARALLEL)
def normalize_weights_inplace(loglik, weights):
    # max や sum は Numba の parallel 下で自動的に Reduction 処理されます
    max_ll = np.max(loglik)
    weights[:] = np.exp(loglik - max_ll)
    w_sum = np.sum(weights)
    
    if w_sum <= 0.0:
        n = weights.shape[0]
        weights[:] = 1.0 / n
    else:
        weights /= w_sum

@njit(**JIT_PARALLEL)
def weighted_mean(particles, weights):
    return np.sum(particles * weights)


# ---- リサンプリング（ここを修正）----

# 【修正】ここは依存関係があるため JIT_SERIAL (parallel=False) に変更
@njit(**JIT_SERIAL)
def systematic_resample_inplace(weights, u, indices):
    n = weights.shape[0]
    cum_weight = weights[0]
    i = 0
    
    for j in range(n):
        target = (j + u) / n
        while i < n - 1 and cum_weight < target:
            i += 1
            cum_weight += weights[i]
        indices[j] = i

# 【維持】コピー処理は独立しているので JIT_PARALLEL のまま (prangeが効く)
@njit(**JIT_PARALLEL)
def apply_resample_inplace(particles, indices, tmp_particles):
    # prangeを使うことで、コピー処理を16スレッドで分担実行します
    for i in prange(particles.shape[0]):
        tmp_particles[i] = particles[indices[i]]
    particles[:] = tmp_particles


# ============================================================
# メインループ
# ============================================================

@njit(**JIT_PARALLEL)
def run_particle_filter_jit(
    y, num_particles, system_id, obs_id, sigma_w, sigma_obs, seed
):
    T = len(y)
    x_hat = np.empty(T)
    eps = 1e-8

    np.random.seed(seed)

    if system_id == 2:
        particles = np.abs(3.0 + np.random.normal(0.0, 1.0, num_particles))
    else:
        particles = np.random.normal(0.0, 1.0, num_particles)

    noise = np.empty(num_particles)
    loglik = np.empty(num_particles)
    weights = np.empty(num_particles)
    indices = np.empty(num_particles, dtype=np.int64)
    tmp_particles = np.empty(num_particles)

    for t in range(T):
        # 1. ノイズ生成
        noise[:] = np.random.normal(0.0, sigma_w, num_particles)

        # 2. 予測
        if system_id == 0:
            predict_linear_inplace(particles, noise)
        elif system_id == 1:
            predict_gordon_inplace(particles, noise, t)
        else:
            predict_positive_rw_inplace(particles, noise, eps)

        # 3. 尤度
        if obs_id == 0:
            loglik_gaussian_inplace(y[t], particles, sigma_obs, loglik)
        else:
            loglik_poisson_inplace(y[t], particles, loglik)

        # 4. 重み & 推定
        normalize_weights_inplace(loglik, weights)
        x_hat[t] = weighted_mean(particles, weights)

        # 5. リサンプリング
        u = np.random.uniform(0.0, 1.0 / num_particles)
        systematic_resample_inplace(weights, u, indices)
        apply_resample_inplace(particles, indices, tmp_particles)

    return x_hat


def run_particle_filter(y, num_particles, system, obs_type, sigma_w, sigma_obs=1.0, seed=0):
    sys_map = {"linear": 0, "gordon": 1, "positive_rw": 2}
    obs_map = {"gaussian": 0, "poisson": 1}
    return run_particle_filter_jit(
        y.astype(np.float64), num_particles, sys_map[system], obs_map[obs_type],
        float(sigma_w), float(sigma_obs), int(seed)
    )

# ============================================================
# ベンチマーク実行用
# ============================================================

def rmse(true, est):
    return float(np.sqrt(np.mean((true - est) ** 2)))

def main():
    try:
        df = pd.read_csv("Data/benchmark_data.csv")
        params_df = pd.read_csv("Data/benchmark_params.csv")
    except FileNotFoundError:
        print("Error: benchmark_data.csv not found. Run generation script first.")
        return

    # 大規模検証用パラメータ
    # 並列化の効果を見るため、粒子数を10^6に設定
    num_particles = 1_000_000 
    n_runs = 1 # 時間がかかるため回数は減らす

    print(f"Python (Numba) Particle Filter Benchmark [LARGE SCALE]")
    print(f"======================================================")
    print(f"Particles : {num_particles:,}")
    print(f"Time Steps: {len(df):,}")
    print(f"Threads   : {get_num_threads()} (via Numba parallel)")
    print(f"Mode      : SIMD + Multi-threading (parallel=True)")
    
    # データ準備
    y1 = df["y1_obs"].values
    y2 = df["y2_obs"].values
    y3 = df["y3_obs"].values.astype(float)
    x1_true = df["x1_true"].values
    x2_true = df["x2_true"].values
    x3_true = df["x3_true"].values

    p1 = params_df[params_df["case"] == "case1"].iloc[0]
    p2 = params_df[params_df["case"] == "case2"].iloc[0]
    p3 = params_df[params_df["case"] == "case3"].iloc[0]

    # Warmup
    print("\nWarming up JIT...")
    # コンパイル時は少量でOK
    run_particle_filter(np.random.randn(100), 1000, "linear", "gaussian", 0.5)
    run_particle_filter(np.random.randn(100), 1000, "gordon", "gaussian", 0.5)
    run_particle_filter(np.abs(np.random.randn(100)), 1000, "positive_rw", "poisson", 0.5)
    print("Warmup complete.\n")

    results = []

    # Case 2: Gordon (計算負荷が高く並列化の恩恵を受けやすい)
    print("Case 2: Gordon Nonlinear + Gaussian")
    times = []
    for r in range(n_runs):
        print(f"  Run {r+1}/{n_runs}...", end="", flush=True)
        t0 = time.perf_counter()
        xh = run_particle_filter(y2, num_particles, "gordon", "gaussian", p2["sigma_w"], p2["sigma_obs"], r)
        dur = time.perf_counter() - t0
        times.append(dur)
        print(f" {dur:.2f}s")
    
    r_val = rmse(x2_true, xh)
    t_mean = np.mean(times)
    t_std = np.std(times)
    print(f"  RMSE: {r_val:.6f}")
    print(f"  Avg Time: {t_mean:.4f} sec\n")
    results.append(["case2", num_particles, t_mean])

    # Case 1 & 3 は省略可能だが、比較のためにCase 1だけ実行してもよい
    print("Case 1: Linear + Gaussian")
    times = []
    for r in range(n_runs):
        print(f"  Run {r+1}/{n_runs}...", end="", flush=True)
        t0 = time.perf_counter()
        xh = run_particle_filter(y1, num_particles, "linear", "gaussian", p1["sigma_w"], p1["sigma_obs"], r)
        dur = time.perf_counter() - t0
        times.append(dur)
        print(f" {dur:.2f}s")
        
    r_val = rmse(x1_true, xh)
    t_mean = np.mean(times)
    print(f"  RMSE: {r_val:.6f}")
    print(f"  Avg Time: {t_mean:.4f} sec\n")
    results.append(["case1", num_particles, t_mean])
    
    # 結果保存
    pd.DataFrame(results, columns=["case", "particles", "time"]).to_csv("Results/python_large.csv", index=False)

if __name__ == "__main__":
    main()