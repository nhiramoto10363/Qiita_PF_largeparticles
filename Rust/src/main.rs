//! 粒子フィルタ Rust実装（シングルスレッド版・Rayon不使用）
//!
//! 言語間ベンチマーク比較用。CSVからデータを読み込み、
//! 3ケースに対して粒子フィルタを実行し、RMSEと計算時間を出力。

use rand::prelude::*;
use rand_distr::{Distribution, Normal, Uniform};
use serde::Deserialize;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

/// ベンチマークデータの1行
#[derive(Debug, Deserialize)]
struct DataRow {
    t: i64,
    x1_true: f64,
    y1_obs: f64,
    x2_true: f64,
    y2_obs: f64,
    x3_true: f64,
    y3_obs: f64,
}

/// パラメータデータの1行
#[derive(Debug, Deserialize)]
struct ParamRow {
    case: String,
    sigma_w: f64,
    sigma_obs: f64,
    system: String,
    obs_type: String,
}

/// システムモデルの種類
#[derive(Clone, Copy)]
enum SystemModel {
    Linear,
    Gordon,
    PositiveRW,
}

/// 観測モデルの種類
#[derive(Clone, Copy)]
enum ObsModel {
    Gaussian,
    Poisson,
}

/// 作業用バッファ（ループ内で使い回す）
struct PFWorkspace {
    noise: Vec<f64>,
    loglik: Vec<f64>,
    weights: Vec<f64>,
    indices: Vec<usize>,
    cumsum: Vec<f64>,
    tmp_particles: Vec<f64>,
}

impl PFWorkspace {
    fn new(n: usize) -> Self {
        Self {
            noise: vec![0.0; n],
            loglik: vec![0.0; n],
            weights: vec![0.0; n],
            indices: vec![0; n],
            cumsum: vec![0.0; n],
            tmp_particles: vec![0.0; n],
        }
    }
}

/// 線形システムの予測（シングルスレッド）
fn predict_linear(particles: &mut [f64], noise: &[f64]) {
    let n = particles.len();
    for i in 0..n {
        particles[i] += noise[i];
    }
}

/// Gordon型非線形システムの予測（シングルスレッド）
fn predict_gordon(particles: &mut [f64], noise: &[f64], t: usize) {
    let forcing = 8.0 * (1.2 * t as f64).cos();
    let n = particles.len();
    for i in 0..n {
        let x = particles[i];
        let nonlinear = 25.0 * x / (1.0 + x * x);
        particles[i] = 0.5 * x + nonlinear + forcing + noise[i];
    }
}

/// 非負ランダムウォークの予測（シングルスレッド）
fn predict_positive_rw(particles: &mut [f64], noise: &[f64], eps: f64) {
    let n = particles.len();
    for i in 0..n {
        let val = particles[i] + noise[i];
        particles[i] = if val > eps { val } else { eps };
    }
}

/// ガウシアン観測の対数尤度（in-place）
fn loglik_gaussian(y: f64, particles: &[f64], sigma: f64, out_loglik: &mut [f64]) {
    let const_term = -0.5 * (2.0 * PI * sigma * sigma).ln();
    let inv_var = 1.0 / (sigma * sigma);

    let n = particles.len();
    for i in 0..n {
        let diff = y - particles[i];
        out_loglik[i] = const_term - 0.5 * diff * diff * inv_var;
    }
}

/// Poisson観測の対数尤度（in-place）
fn loglik_poisson(y: f64, particles: &[f64], out_loglik: &mut [f64]) {
    let log_gamma_y = ln_gamma(y + 1.0);
    let n = particles.len();

    for i in 0..n {
        let mut lam = particles[i];
        if lam < 1e-8 {
            lam = 1e-8;
        }
        out_loglik[i] = y * lam.ln() - lam - log_gamma_y;
    }
}

/// log-gamma関数（Lanczos近似）
fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation coefficients
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        let z = 1.0 - x;
        PI.ln() - (PI * x).sin().ln() - ln_gamma(z)
    } else {
        let z = x - 1.0;
        let mut sum = C[0];
        for i in 1..9 {
            sum += C[i] / (z + i as f64);
        }
        let t = z + G + 0.5;
        0.5 * (2.0 * PI).ln() + (t.ln() * (z + 0.5)) - t + sum.ln()
    }
}

/// 重みの正規化（数値安定化, in-place）
fn normalize_weights(loglik: &[f64], weights: &mut [f64]) {
    let n = loglik.len();
    let mut max_ll = f64::NEG_INFINITY;
    for i in 0..n {
        if loglik[i] > max_ll {
            max_ll = loglik[i];
        }
    }

    // weights[i] = exp(loglik[i] - max_ll)
    let mut sum = 0.0;
    for i in 0..n {
        let w = (loglik[i] - max_ll).exp();
        weights[i] = w;
        sum += w;
    }

    if sum <= 0.0 {
        let val = 1.0 / n as f64;
        for w in weights.iter_mut() {
            *w = val;
        }
    } else {
        let inv_sum = 1.0 / sum;
        for w in weights.iter_mut() {
            *w *= inv_sum;
        }
    }
}

/// 重み付き平均
fn weighted_mean(particles: &[f64], weights: &[f64]) -> f64 {
    let n = particles.len();
    let mut s = 0.0;
    for i in 0..n {
        s += particles[i] * weights[i];
    }
    s
}

/// システマティックリサンプリング（cumsum バッファを再利用）
fn systematic_resample(
    weights: &[f64],
    u: f64,
    indices: &mut [usize],
    cumsum: &mut [f64],
) {
    let n = weights.len();

    cumsum[0] = weights[0];
    for i in 1..n {
        cumsum[i] = cumsum[i - 1] + weights[i];
    }

    let mut i = 0usize;
    for j in 0..n {
        let target = (j as f64 + u) / n as f64;
        while i < n - 1 && cumsum[i] < target {
            i += 1;
        }
        indices[j] = i;
    }
}

/// リサンプリングの適用（tmp バッファを再利用）
fn apply_resample(particles: &mut [f64], indices: &[usize], tmp: &mut [f64]) {
    let n = particles.len();
    for i in 0..n {
        tmp[i] = particles[indices[i]];
    }
    for i in 0..n {
        particles[i] = tmp[i];
    }
}

/// 粒子フィルタのメインループ（シングルスレッド版）
fn run_particle_filter(
    y: &[f64],
    num_particles: usize,
    system: SystemModel,
    obs_model: ObsModel,
    sigma_w: f64,
    sigma_obs: f64,
    seed: u64,
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal_proc = Normal::new(0.0, sigma_w).unwrap();
    let normal_init = Normal::new(0.0, 1.0).unwrap();
    let uniform = Uniform::new(0.0, 1.0 / num_particles as f64);

    let t_len = y.len();
    let eps = 1e-8;

    // 初期化
    let mut particles: Vec<f64> = match system {
        SystemModel::PositiveRW => {
            let mut v = Vec::with_capacity(num_particles);
            for _ in 0..num_particles {
                let x: f64 = 3.0_f64 + normal_init.sample(&mut rng);
                v.push(x.abs());
            }
            v
        }
        _ => {
            let mut v = Vec::with_capacity(num_particles);
            for _ in 0..num_particles {
                v.push(normal_init.sample(&mut rng));
            }
            v
        }
    };

    // 作業バッファ
    let mut work = PFWorkspace::new(num_particles);

    let mut x_hat = vec![0.0; t_len];

    for t in 0..t_len {
        // プロセスノイズ生成
        for i in 0..num_particles {
            work.noise[i] = normal_proc.sample(&mut rng);
        }

        // 予測ステップ
        match system {
            SystemModel::Linear => predict_linear(&mut particles, &work.noise),
            SystemModel::Gordon => predict_gordon(&mut particles, &work.noise, t),
            SystemModel::PositiveRW => predict_positive_rw(&mut particles, &work.noise, eps),
        }

        // 尤度計算
        match obs_model {
            ObsModel::Gaussian => {
                loglik_gaussian(y[t], &particles, sigma_obs, &mut work.loglik);
            }
            ObsModel::Poisson => {
                loglik_poisson(y[t], &particles, &mut work.loglik);
            }
        }

        // 重み正規化
        normalize_weights(&work.loglik, &mut work.weights);

        // 状態推定値
        x_hat[t] = weighted_mean(&particles, &work.weights);

        // リサンプリング
        let u = uniform.sample(&mut rng);
        systematic_resample(
            &work.weights,
            u,
            &mut work.indices,
            &mut work.cumsum,
        );
        apply_resample(&mut particles, &work.indices, &mut work.tmp_particles);
    }

    x_hat
}

/// RMSE計算
fn rmse(true_vals: &[f64], est_vals: &[f64]) -> f64 {
    let n = true_vals.len();
    let mut mse = 0.0;
    for i in 0..n {
        let d = true_vals[i] - est_vals[i];
        mse += d * d;
    }
    (mse / n as f64).sqrt()
}

/// CSVデータ読み込み
fn load_data(path: &str) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let file = File::open(path).expect("Cannot open data file");
    let mut reader = csv::Reader::from_reader(file);

    let mut x1_true = Vec::new();
    let mut y1_obs = Vec::new();
    let mut x2_true = Vec::new();
    let mut y2_obs = Vec::new();
    let mut x3_true = Vec::new();
    let mut y3_obs = Vec::new();

    for result in reader.deserialize() {
        let row: DataRow = result.expect("Cannot parse row");
        x1_true.push(row.x1_true);
        y1_obs.push(row.y1_obs);
        x2_true.push(row.x2_true);
        y2_obs.push(row.y2_obs);
        x3_true.push(row.x3_true);
        y3_obs.push(row.y3_obs);
    }

    (x1_true, y1_obs, x2_true, y2_obs, x3_true, y3_obs)
}

/// パラメータ読み込み
fn load_params(path: &str) -> Vec<ParamRow> {
    let file = File::open(path).expect("Cannot open params file");
    let mut reader = csv::Reader::from_reader(file);

    reader
        .deserialize()
        .map(|r| r.expect("Cannot parse param row"))
        .collect()
}

fn main() {
    // データ読み込み
    let (x1_true, y1_obs, x2_true, y2_obs, x3_true, y3_obs) =
        load_data("../Data/benchmark_data.csv");
    let params = load_params("../Data/benchmark_params.csv");

    let num_particles = 1000000;
    let n_runs = 1;

    println!("Rust Particle Filter Benchmark (single-thread, no Rayon)");
    println!("======================================================\n");

    let mut results = Vec::new();

    // ケース1: 線形 + ガウシアン
    println!("Case 1: Linear + Gaussian");
    let p1 = &params[0];
    let mut times1 = Vec::new();
    let mut xhat1 = Vec::new();

    for run in 0..n_runs {
        let start = Instant::now();
        xhat1 = run_particle_filter(
            &y1_obs,
            num_particles,
            SystemModel::Linear,
            ObsModel::Gaussian,
            p1.sigma_w,
            p1.sigma_obs,
            run as u64,
        );
        times1.push(start.elapsed().as_secs_f64());
    }

    let rmse1 = rmse(&x1_true, &xhat1);
    let mean_time1: f64 = times1.iter().sum::<f64>() / n_runs as f64;
    let std_time1: f64 = (times1.iter().map(|t| (t - mean_time1).powi(2)).sum::<f64>()
        / n_runs as f64)
        .sqrt();
    println!("  RMSE: {:.6}", rmse1);
    println!("  Time: {:.4} ± {:.4} sec\n", mean_time1, std_time1);
    results.push((
        "case1_linear_gaussian",
        rmse1,
        mean_time1,
        std_time1,
    ));

    // ケース2: Gordon非線形 + ガウシアン
    println!("Case 2: Gordon Nonlinear + Gaussian");
    let p2 = &params[1];
    let mut times2 = Vec::new();
    let mut xhat2 = Vec::new();

    for run in 0..n_runs {
        let start = Instant::now();
        xhat2 = run_particle_filter(
            &y2_obs,
            num_particles,
            SystemModel::Gordon,
            ObsModel::Gaussian,
            p2.sigma_w,
            p2.sigma_obs,
            run as u64,
        );
        times2.push(start.elapsed().as_secs_f64());
    }

    let rmse2 = rmse(&x2_true, &xhat2);
    let mean_time2: f64 = times2.iter().sum::<f64>() / n_runs as f64;
    let std_time2: f64 = (times2.iter().map(|t| (t - mean_time2).powi(2)).sum::<f64>()
        / n_runs as f64)
        .sqrt();
    println!("  RMSE: {:.6}", rmse2);
    println!("  Time: {:.4} ± {:.4} sec\n", mean_time2, std_time2);
    results.push((
        "case2_gordon_nonlinear",
        rmse2,
        mean_time2,
        std_time2,
    ));

    // ケース3: 非負RW + Poisson
    println!("Case 3: Positive RW + Poisson");
    let p3 = &params[2];
    let mut times3 = Vec::new();
    let mut xhat3 = Vec::new();

    for run in 0..n_runs {
        let start = Instant::now();
        xhat3 = run_particle_filter(
            &y3_obs,
            num_particles,
            SystemModel::PositiveRW,
            ObsModel::Poisson,
            p3.sigma_w,
            0.0,
            run as u64,
        );
        times3.push(start.elapsed().as_secs_f64());
    }

    let rmse3 = rmse(&x3_true, &xhat3);
    let mean_time3: f64 = times3.iter().sum::<f64>() / n_runs as f64;
    let std_time3: f64 = (times3.iter().map(|t| (t - mean_time3).powi(2)).sum::<f64>()
        / n_runs as f64)
        .sqrt();
    println!("  RMSE: {:.6}", rmse3);
    println!("  Time: {:.4} ± {:.4} sec\n", mean_time3, std_time3);
    results.push((
        "case3_positive_rw_poisson",
        rmse3,
        mean_time3,
        std_time3,
    ));

    // 結果をCSV出力
    let mut file = File::create("../results_rust_single.csv").expect("Cannot create results file");
    writeln!(file, "case,language,num_particles,rmse,time_mean_sec,time_std_sec").unwrap();
    for (case, rmse_val, mean_time, std_time) in &results {
        writeln!(
            file,
            "{},Rust (single-thread),{},{:.6},{:.6},{:.6}",
            case, num_particles, rmse_val, mean_time, std_time
        )
        .unwrap();
    }

    println!("Saved: results_rust_single.csv");
}
