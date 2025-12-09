#=
pf_julia.jl
粒子フィルタ Julia実装（マルチスレッド + SIMD最適化版） - 大規模検証用

実行方法:
  julia --threads=auto Julia/pf_julia.jl

必ず --threads オプションをつけて実行してください。
=#

using Random
using Statistics
using SpecialFunctions: loggamma
using CSV
using DataFrames
using Base.Threads

# ============================================================
# 予測ステップ（並列化）
# ============================================================

function predict_linear!(particles::Vector{Float64}, noise::Vector{Float64})
    # スレッド並列化により、100万粒子を一気に更新
    @threads for i in eachindex(particles)
        @inbounds particles[i] += noise[i]
    end
end

function predict_gordon!(particles::Vector{Float64}, noise::Vector{Float64}, t::Int)
    forcing = 8.0 * cos(1.2 * t)
    @threads for i in eachindex(particles)
        @inbounds begin
            x = particles[i]
            nonlinear = 25.0 * x / (1.0 + x * x)
            particles[i] = 0.5 * x + nonlinear + forcing + noise[i]
        end
    end
end

function predict_positive_rw!(particles::Vector{Float64}, noise::Vector{Float64}, eps::Float64)
    @threads for i in eachindex(particles)
        @inbounds begin
            val = particles[i] + noise[i]
            particles[i] = val > eps ? val : eps
        end
    end
end

# ============================================================
# 尤度計算（並列化）
# ============================================================

function loglik_gaussian!(loglik::Vector{Float64}, y::Float64, particles::Vector{Float64}, sigma::Float64)
    const_term = -0.5 * log(2π * sigma^2)
    inv_var = 1.0 / (sigma^2)
    
    @threads for i in eachindex(particles)
        @inbounds begin
            diff = y - particles[i]
            loglik[i] = const_term - 0.5 * diff^2 * inv_var
        end
    end
end

function loglik_poisson!(loglik::Vector{Float64}, y::Float64, particles::Vector{Float64})
    log_gamma_y = loggamma(y + 1.0)
    
    @threads for i in eachindex(particles)
        @inbounds begin
            lam = max(particles[i], 1e-8)
            loglik[i] = y * log(lam) - lam - log_gamma_y
        end
    end
end

# ============================================================
# 重み正規化（一部並列化）
# ============================================================

function normalize_weights!(weights::Vector{Float64}, loglik::Vector{Float64})
    # maximumやsumは標準関数の最適化に任せる（並列化ライブラリを使わない限りシングルスレッドだが高速）
    max_ll = maximum(loglik)
    
    # exp計算は並列化
    @threads for i in eachindex(weights)
        @inbounds weights[i] = exp(loglik[i] - max_ll)
    end
    
    w_sum = sum(weights)
    
    if w_sum <= 0.0
        fill!(weights, 1.0 / length(weights))
    else
        # 正規化割り算も並列化
        @threads for i in eachindex(weights)
            @inbounds weights[i] /= w_sum
        end
    end
end

# ============================================================
# 重み付き平均
# ============================================================

function weighted_mean(particles::Vector{Float64}, weights::Vector{Float64})
    @assert length(particles) == length(weights)
    s = 0.0
    @inbounds @simd for i in eachindex(particles, weights)
        s += particles[i] * weights[i]
    end
    return s
end


# ============================================================
# システマティックリサンプリング（直列）
# ============================================================
# ※ここは依存関係が強く並列化困難なため、Python版同様にシングルスレッドで実行

function systematic_resample!(indices::Vector{Int}, weights::Vector{Float64}, u::Float64)
    n = length(weights)
    
    # 累積和
    cumsum_w = cumsum(weights)
    
    # リサンプリング
    j = 1
    for i in 1:n
        target = (i - 1 + u) / n
        while j < n && cumsum_w[j] < target
            j += 1
        end
        indices[i] = j
    end
end

# ============================================================
# リサンプリング適用（並列化）
# ============================================================
# ※メモリコピー処理は並列化の効果が高い

function apply_resample!(particles::Vector{Float64}, particles_new::Vector{Float64}, indices::Vector{Int})
    @threads for i in eachindex(particles)
        @inbounds particles_new[i] = particles[indices[i]]
    end
    copyto!(particles, particles_new)
end

# ============================================================
# 粒子フィルタ メインループ
# ============================================================

function run_particle_filter(
    y::Vector{Float64},
    num_particles::Int,
    system::Symbol,  # :linear, :gordon, :positive_rw
    obs_type::Symbol,  # :gaussian, :poisson
    sigma_w::Float64,
    sigma_obs::Float64,
    seed::Int
)
    rng = MersenneTwister(seed)
    T = length(y)
    eps = 1e-8
    
    # 初期化
    particles = if system == :positive_rw
        abs.(3.0 .+ randn(rng, num_particles))
    else
        randn(rng, num_particles)
    end
    
    # 作業用配列
    noise = zeros(num_particles)
    loglik = zeros(num_particles)
    weights = zeros(num_particles)
    indices = zeros(Int, num_particles)
    particles_new = zeros(num_particles)
    
    x_hat = zeros(T)
    
    for t in 1:T
        # プロセスノイズ生成
        # randn!自体はシングルスレッドだが、BLAS最適化されており高速。
        # 必要ならTaskLocalRNG等で並列化できるが、今回は計算負荷の検証が主眼なので標準を使用。
        randn!(rng, noise)
        
        # ノイズのスケーリング（並列化）
        @threads for i in 1:num_particles
            @inbounds noise[i] *= sigma_w
        end
        
        # 予測ステップ
        if system == :linear
            predict_linear!(particles, noise)
        elseif system == :gordon
            predict_gordon!(particles, noise, t - 1)
        elseif system == :positive_rw
            predict_positive_rw!(particles, noise, eps)
        end
        
        # 尤度計算
        if obs_type == :gaussian
            loglik_gaussian!(loglik, y[t], particles, sigma_obs)
        elseif obs_type == :poisson
            loglik_poisson!(loglik, y[t], particles)
        end
        
        # 重み正規化
        normalize_weights!(weights, loglik)
        
        # 状態推定値
        x_hat[t] = weighted_mean(particles, weights)
        
        # リサンプリング
        u = rand(rng) / num_particles
        systematic_resample!(indices, weights, u)
        apply_resample!(particles, particles_new, indices)
    end
    
    return x_hat
end

# ============================================================
# RMSE計算
# ============================================================

function rmse(true_vals::Vector{Float64}, est_vals::Vector{Float64})
    sqrt(mean((true_vals .- est_vals).^2))
end

# ============================================================
# ベンチマーク実行
# ============================================================

function main()
    # データ読み込み
    data_path = "Data/benchmark_data.csv"
    params_path = "Data/benchmark_params.csv"
    
    if !isfile(data_path)
        println("Error: $data_path not found.")
        println("Please run the Python data generation script with --T 50000 first.")
        return
    end

    data = CSV.read(data_path, DataFrame)
    params = CSV.read(params_path, DataFrame)
    
    y1 = Vector{Float64}(data.y1_obs)
    y2 = Vector{Float64}(data.y2_obs)
    y3 = Vector{Float64}(data.y3_obs)
    
    x1_true = Vector{Float64}(data.x1_true)
    x2_true = Vector{Float64}(data.x2_true)
    x3_true = Vector{Float64}(data.x3_true)
    
    # 大規模設定
    num_particles = 1_000_000
    n_runs = 1
    
    println("Julia Particle Filter Benchmark [LARGE SCALE]")
    println("=============================================")
    println("Particles : ", num_particles)
    println("Time Steps: ", length(y1))
    println("Threads   : ", Threads.nthreads())
    
    if Threads.nthreads() == 1
        println("WARNING: Running with 1 thread. Use 'julia --threads=auto' for parallel speedup.")
    end
    println()
    
    results = DataFrame(
        case = String[],
        particles = Int[],
        time = Float64[]
    )
    
    # ウォームアップ
    println("Warming up JIT compilation...")
    run_particle_filter(y1[1:100], 1000, :linear, :gaussian, 0.5, 1.0, 999)
    run_particle_filter(y2[1:100], 1000, :gordon, :gaussian, 0.3, 1.0, 999)
    run_particle_filter(y3[1:100], 1000, :positive_rw, :poisson, 0.2, 0.0, 999)
    println("Warmup complete.")
    println()
    
    # Case 2: Gordon (計算負荷大)
    println("Case 2: Gordon Nonlinear + Gaussian")
    sigma_w2 = params[2, :sigma_w]
    sigma_obs2 = params[2, :sigma_obs]
    times2 = Float64[]
    local xhat2
    
    for run in 1:n_runs
        print("  Run $run/$n_runs...")
        t = @elapsed xhat2 = run_particle_filter(y2, num_particles, :gordon, :gaussian, sigma_w2, sigma_obs2, run)
        push!(times2, t)
        println(" $(round(t, digits=2))s")
    end
    
    rmse2 = rmse(x2_true, xhat2)
    mean_time2 = mean(times2)
    println("  RMSE: ", round(rmse2, digits=6))
    println("  Avg Time: ", round(mean_time2, digits=4), " sec")
    println()
    push!(results, ("case2", num_particles, mean_time2))
    
    # Case 1: Linear
    println("Case 1: Linear + Gaussian")
    sigma_w1 = params[1, :sigma_w]
    sigma_obs1 = params[1, :sigma_obs]
    times1 = Float64[]
    local xhat1
    
    for run in 1:n_runs
        print("  Run $run/$n_runs...")
        t = @elapsed xhat1 = run_particle_filter(y1, num_particles, :linear, :gaussian, sigma_w1, sigma_obs1, run)
        push!(times1, t)
        println(" $(round(t, digits=2))s")
    end
    
    rmse1 = rmse(x1_true, xhat1)
    mean_time1 = mean(times1)
    println("  RMSE: ", round(rmse1, digits=6))
    println("  Avg Time: ", round(mean_time1, digits=4), " sec")
    println()
    push!(results, ("case1", num_particles, mean_time1))
    
    # 結果出力
    CSV.write("Results/julia_large.csv", results)
    println("Saved: Results/julia_large.csv")
end

main()