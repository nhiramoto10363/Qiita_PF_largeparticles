"""
粒子フィルタ言語比較用データ生成スクリプト（大規模対応版）

変更点:
- argparse導入により、コマンドラインからデータ長 T を指定可能に。
- デフォルト値を T=50,000 (大規模) に設定。
"""

import numpy as np
import pandas as pd
import argparse
import os

def generate_case1_linear_gaussian(T, seed=0, sigma_w=0.5, sigma_obs=1.0, x0=0.0):
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    x[0] = x0
    # 状態遷移（ベクトル化はできないためループ）
    w = rng.normal(0.0, sigma_w, size=T)
    for t in range(1, T):
        x[t] = x[t - 1] + w[t]
    
    y = x + rng.normal(0.0, sigma_obs, size=T)
    return x, y

def generate_case2_gordon_nonlinear(T, seed=1, sigma_w=0.3, sigma_obs=1.0, x0=0.0):
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    x[0] = x0
    w = rng.normal(0.0, sigma_w, size=T)
    
    for t in range(1, T):
        nonlinear = 25.0 * x[t - 1] / (1.0 + x[t - 1] ** 2)
        forcing = 8.0 * np.cos(1.2 * t)
        x[t] = 0.5 * x[t - 1] + nonlinear + forcing + w[t]

    y = x + rng.normal(0.0, sigma_obs, size=T)
    return x, y

def generate_case3_poisson(T, seed=2, sigma_w=0.2, x0=3.0, eps=1e-3):
    rng = np.random.default_rng(seed)
    lam = np.empty(T)
    lam[0] = max(x0, eps)
    w = rng.normal(0.0, sigma_w, size=T)

    for t in range(1, T):
        lam_prop = lam[t - 1] + w[t]
        lam[t] = max(lam_prop, eps)

    y = rng.poisson(lam=lam)
    return lam, y

def main():
    parser = argparse.ArgumentParser(description="Generate Benchmark Data")
    parser.add_argument("--T", type=int, default=50000, help="Time steps (default: 50,000)")
    args = parser.parse_args()

    T = args.T
    print(f"Generating data with T={T}...")

    # ディレクトリ作成
    os.makedirs("Data", exist_ok=True)

    # パラメータ設定
    params = {
        "case1": {"sigma_w": 0.5, "sigma_obs": 1.0},
        "case2": {"sigma_w": 0.3, "sigma_obs": 1.0},
        "case3": {"sigma_w": 0.2, "x0": 3.0},
    }

    # データ生成
    x1, y1 = generate_case1_linear_gaussian(T=T, **params["case1"])
    x2, y2 = generate_case2_gordon_nonlinear(T=T, **params["case2"])
    x3, y3 = generate_case3_poisson(T=T, **params["case3"])

    # CSV出力
    df = pd.DataFrame({
        "t": np.arange(T),
        "x1_true": x1, "y1_obs": y1,
        "x2_true": x2, "y2_obs": y2,
        "x3_true": x3, "y3_obs": y3,
    })
    df.to_csv("Data/benchmark_data.csv", index=False)

    # パラメータ保存
    params_df = pd.DataFrame({
        "case": ["case1", "case2", "case3"],
        "sigma_w": [params["case1"]["sigma_w"], params["case2"]["sigma_w"], params["case3"]["sigma_w"]],
        "sigma_obs": [params["case1"]["sigma_obs"], params["case2"]["sigma_obs"], 0.0],
        "system": ["linear", "gordon", "positive_rw"],
        "obs_type": ["gaussian", "gaussian", "poisson"],
    })
    params_df.to_csv("Data/benchmark_params.csv", index=False)

    print(f"Done. Saved to Data/ directory.")

if __name__ == "__main__":
    main()