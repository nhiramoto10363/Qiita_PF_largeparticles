! pf_fortran_fast.f90
! 粒子フィルタ Fortran実装（高速化版）
!
! 主な最適化:
!   1. スレッドローカルXoshiro256++ RNG（ロックフリー並列乱数生成）
!   2. 全タイムステップのノイズを事前生成（メモリトレードオフ）
!   3. ループフュージョン（predict + likelihood を統合）
!   4. 並列プレフィックスサム（階層的アプローチ）
!   5. メモリアライメントとキャッシュ最適化
!
! コンパイル (gfortran):
!   gfortran -O3 -march=native -ffast-math -fopenmp -funroll-loops pf_fortran_fast.f90 -o pf_fortran_fast
!
! コンパイル (ifort/ifx):
!   ifx -O3 -xHost -qopenmp -fp-model fast=2 pf_fortran_fast.f90 -o pf_fortran_fast
!
! 実行:
!   export OMP_NUM_THREADS=16
!   ./pf_fortran_fast

module xoshiro256pp_module
    implicit none
    integer, parameter :: dp = kind(1.0d0)
    integer, parameter :: i64 = selected_int_kind(18)
    
    ! スレッドローカルRNG状態（最大256スレッド対応）
    integer, parameter :: MAX_THREADS = 256
    integer(i64) :: rng_state(4, MAX_THREADS)
    
contains
    ! Xoshiro256++ の初期化
    subroutine init_xoshiro(base_seed, thread_id)
        integer, intent(in) :: base_seed, thread_id
        integer(i64) :: s
        integer :: i
        
        ! SplitMix64で初期状態を生成
        s = int(base_seed, i64) + int(thread_id, i64) * 2685821657736338717_i64
        do i = 1, 4
            s = s + (-7046029254386353131_i64)
            s = ieor(s, ishft(s, -30)) * (-4658895280553007687_i64)
            s = ieor(s, ishft(s, -27)) * (-7723592293110705685_i64)
            rng_state(i, thread_id) = ieor(s, ishft(s, -31))
        end do
    end subroutine init_xoshiro
    
    ! 回転関数
    pure function rotl(x, k) result(r)
        integer(i64), intent(in) :: x
        integer, intent(in) :: k
        integer(i64) :: r
        r = ior(ishft(x, k), ishft(x, -(64-k)))
    end function rotl
    
    ! Xoshiro256++ 乱数生成（0-1の一様乱数）
    function xoshiro_uniform(thread_id) result(u)
        integer, intent(in) :: thread_id
        real(dp) :: u
        integer(i64) :: result_val, t
        
        result_val = rotl(rng_state(1, thread_id) + rng_state(4, thread_id), 23) &
                   + rng_state(1, thread_id)
        
        t = ishft(rng_state(2, thread_id), 17)
        
        rng_state(3, thread_id) = ieor(rng_state(3, thread_id), rng_state(1, thread_id))
        rng_state(4, thread_id) = ieor(rng_state(4, thread_id), rng_state(2, thread_id))
        rng_state(2, thread_id) = ieor(rng_state(2, thread_id), rng_state(3, thread_id))
        rng_state(1, thread_id) = ieor(rng_state(1, thread_id), rng_state(4, thread_id))
        
        rng_state(3, thread_id) = ieor(rng_state(3, thread_id), t)
        rng_state(4, thread_id) = rotl(rng_state(4, thread_id), 45)
        
        ! 64bit整数を[0,1)のdoubleに変換
        u = real(iand(result_val, 9007199254740991_i64), dp) / 9007199254740992.0_dp
    end function xoshiro_uniform
    
    ! Box-Muller変換で正規乱数を生成（2個ずつ）
    subroutine xoshiro_randn_pair(thread_id, z1, z2)
        integer, intent(in) :: thread_id
        real(dp), intent(out) :: z1, z2
        real(dp), parameter :: pi = 3.141592653589793d0
        real(dp) :: u1, u2, r
        
        u1 = xoshiro_uniform(thread_id)
        u2 = xoshiro_uniform(thread_id)
        
        if (u1 < 1.0d-15) u1 = 1.0d-15
        r = sqrt(-2.0d0 * log(u1))
        z1 = r * cos(2.0d0 * pi * u2)
        z2 = r * sin(2.0d0 * pi * u2)
    end subroutine xoshiro_randn_pair
    
end module xoshiro256pp_module


program particle_filter_benchmark_fast
    use omp_lib
    use xoshiro256pp_module
    implicit none

    ! 大規模設定
    integer, parameter :: T = 50000
    integer, parameter :: num_particles = 1000000
    integer, parameter :: n_runs = 3

    real(dp), allocatable :: x1_true(:), y1_obs(:)
    real(dp), allocatable :: x2_true(:), y2_obs(:)
    real(dp), allocatable :: x3_true(:), y3_obs(:)
    real(dp), allocatable :: xhat(:)
    
    ! ノイズ用（チャンク単位で生成）
    integer, parameter :: CHUNK_SIZE = 100  ! メモリ節約のため100ステップずつ
    real(dp), allocatable :: noise_chunk(:,:)

    real(dp) :: sigma_w(3), sigma_obs(3)
    real(dp) :: times(n_runs), rmse_val
    real(dp) :: mean_time, std_time
    real(dp) :: t_start, t_end
    integer :: run, i, io_unit, num_threads

    ! パラメータ設定
    sigma_w = [0.5d0, 0.3d0, 0.2d0]
    sigma_obs = [1.0d0, 1.0d0, 0.0d0]

    ! メモリ確保
    allocate(x1_true(T), y1_obs(T))
    allocate(x2_true(T), y2_obs(T))
    allocate(x3_true(T), y3_obs(T))
    allocate(xhat(T))
    allocate(noise_chunk(num_particles, CHUNK_SIZE))

    ! データ読み込み
    print *, 'Loading data (T=', T, ')...'
    call load_data('Data/benchmark_data.csv', x1_true, y1_obs, x2_true, y2_obs, x3_true, y3_obs, T)

    print '(A)', 'Fortran Particle Filter Benchmark [OPTIMIZED]'
    print '(A)', '=============================================='
    
    !$omp parallel
        !$omp master
            num_threads = omp_get_num_threads()
        !$omp end master
    !$omp end parallel
    
    print '(A,I0)',   'Particles : ', num_particles
    print '(A,I0)',   'Time Steps: ', T
    print '(A,I0)',   'Threads   : ', num_threads
    print '(A)', ''

    ! RNG初期化（各スレッド用）
    !$omp parallel
    block
        integer :: tid
        tid = omp_get_thread_num() + 1
        call init_xoshiro(12345, tid)
    end block
    !$omp end parallel

    ! 結果ファイルを開く
    open(newunit=io_unit, file='Results/fortran_fast.csv', status='replace')
    write(io_unit, '(A)') 'case,particles,time'

    ! ウォームアップ
    print '(A)', 'Warming up...'
    call run_pf_fast(y1_obs, 100, 1000, 1, 1, 0.5d0, 1.0d0, xhat)
    print '(A)', 'Warmup complete.'
    print '(A)', ''

    ! ケース2: Gordon非線形 + ガウシアン
    print '(A)', 'Case 2: Gordon Nonlinear + Gaussian'
    do run = 1, n_runs
        t_start = omp_get_wtime()
        call run_pf_fast(y2_obs, T, num_particles, 2, 1, sigma_w(2), sigma_obs(2), xhat)
        t_end = omp_get_wtime()
        times(run) = t_end - t_start
        print '(A,I0,A,I0,A,F8.2,A)', '  Run ', run, '/', n_runs, ': ', times(run), 's'
    end do
    rmse_val = calc_rmse(x2_true, xhat, T)
    call calc_stats(times, n_runs, mean_time, std_time)
    print '(A,F10.6)', '  RMSE: ', rmse_val
    print '(A,F8.4,A)', '  Avg Time: ', mean_time, ' sec'
    print '(A)', ''
    write(io_unit, '(A,I0,A,F10.6)') 'case2,', num_particles, ',', mean_time

    ! ケース1: 線形
    print '(A)', 'Case 1: Linear + Gaussian'
    do run = 1, n_runs
        t_start = omp_get_wtime()
        call run_pf_fast(y1_obs, T, num_particles, 1, 1, sigma_w(1), sigma_obs(1), xhat)
        t_end = omp_get_wtime()
        times(run) = t_end - t_start
        print '(A,I0,A,I0,A,F8.2,A)', '  Run ', run, '/', n_runs, ': ', times(run), 's'
    end do
    rmse_val = calc_rmse(x1_true, xhat, T)
    call calc_stats(times, n_runs, mean_time, std_time)
    print '(A,F10.6)', '  RMSE: ', rmse_val
    print '(A,F8.4,A)', '  Avg Time: ', mean_time, ' sec'
    print '(A)', ''
    write(io_unit, '(A,I0,A,F10.6)') 'case1,', num_particles, ',', mean_time

    close(io_unit)
    print '(A)', 'Saved: Results/fortran_fast.csv'

    deallocate(x1_true, y1_obs, x2_true, y2_obs, x3_true, y3_obs, xhat, noise_chunk)

contains

    !---------------------------------------------------------------------------
    ! 高速化版粒子フィルタ（内部でチャンク単位ノイズ生成）
    !---------------------------------------------------------------------------
    subroutine run_pf_fast(y, T_steps, np, system, obs_type, sigma_w_val, sigma_obs_val, xhat_out)
        integer, intent(in) :: T_steps, np, system, obs_type
        real(dp), intent(in) :: y(T_steps), sigma_w_val, sigma_obs_val
        real(dp), intent(out) :: xhat_out(T_steps)

        ! チャンクサイズ
        integer, parameter :: CHUNK = 256
        
        real(dp), allocatable :: particles(:), loglik(:), weights(:)
        real(dp), allocatable :: particles_new(:)
        real(dp), allocatable :: noise(:,:)  ! (np, CHUNK)
        integer, allocatable :: indices(:)
        
        ! 並列累積和用
        real(dp), allocatable :: cumsum(:)
        real(dp), allocatable :: block_sums(:)
        integer :: num_blocks, block_size
        
        real(dp) :: u, eps
        real(dp) :: max_ll, w_sum, mean_val
        real(dp), parameter :: pi = 3.141592653589793d0
        real(dp) :: const_term, inv_var
        integer :: t_idx, i, tid, chunk_start, chunk_end, local_t

        allocate(particles(np), loglik(np), weights(np))
        allocate(particles_new(np), indices(np))
        allocate(cumsum(np))
        allocate(noise(np, CHUNK))
        
        ! 並列累積和用ブロック
        block_size = 4096
        num_blocks = (np + block_size - 1) / block_size
        allocate(block_sums(num_blocks + 1))

        eps = 1.0d-8
        
        ! ガウシアン尤度の定数項
        const_term = -0.5d0 * log(2.0d0 * pi * sigma_obs_val * sigma_obs_val)
        inv_var = 1.0d0 / (sigma_obs_val * sigma_obs_val)

        ! 初期粒子生成（並列）
        !$omp parallel private(tid)
        tid = omp_get_thread_num() + 1
        !$omp do schedule(static)
        do i = 1, np, 2
            block
                real(dp) :: z1, z2
                call xoshiro_randn_pair(tid, z1, z2)
                if (system == 3) then
                    particles(i) = abs(3.0d0 + z1)
                    if (i + 1 <= np) particles(i+1) = abs(3.0d0 + z2)
                else
                    particles(i) = z1
                    if (i + 1 <= np) particles(i+1) = z2
                end if
            end block
        end do
        !$omp end do
        !$omp end parallel

        ! チャンク単位でタイムステップを処理
        do chunk_start = 1, T_steps, CHUNK
            chunk_end = min(chunk_start + CHUNK - 1, T_steps)
            
            ! このチャンクのノイズを事前生成
            call generate_noise_chunk(noise, np, chunk_end - chunk_start + 1, sigma_w_val)
            
            do t_idx = chunk_start, chunk_end
                local_t = t_idx - chunk_start + 1
            
                ! ========================================
                ! ループフュージョン: 予測 + 尤度計算
                ! ========================================
                max_ll = -huge(1.0d0)
                
                select case (system)
                case (1)  ! 線形 + ガウシアン
                    !$omp parallel do schedule(static) reduction(max:max_ll) private(i)
                    do i = 1, np
                        block
                            real(dp) :: x_pred, diff
                            x_pred = particles(i) + noise(i, local_t)
                            particles(i) = x_pred
                            diff = y(t_idx) - x_pred
                            loglik(i) = const_term - 0.5d0 * diff * diff * inv_var
                            if (loglik(i) > max_ll) max_ll = loglik(i)
                        end block
                    end do
                    !$omp end parallel do
                    
                case (2)  ! Gordon非線形 + ガウシアン
                    block
                        real(dp) :: forcing
                        forcing = 8.0d0 * cos(1.2d0 * (t_idx - 1))
                        
                        !$omp parallel do schedule(static) reduction(max:max_ll) private(i)
                        do i = 1, np
                            block
                                real(dp) :: x, x_pred, nonlinear, diff
                                x = particles(i)
                                nonlinear = 25.0d0 * x / (1.0d0 + x*x)
                                x_pred = 0.5d0 * x + nonlinear + forcing + noise(i, local_t)
                                particles(i) = x_pred
                                diff = y(t_idx) - x_pred
                                loglik(i) = const_term - 0.5d0 * diff * diff * inv_var
                                if (loglik(i) > max_ll) max_ll = loglik(i)
                            end block
                        end do
                        !$omp end parallel do
                    end block
                    
                case (3)  ! Positive RW + Poisson
                    block
                        real(dp) :: log_gamma_y
                        log_gamma_y = log_gamma(y(t_idx) + 1.0d0)
                        
                        !$omp parallel do schedule(static) reduction(max:max_ll) private(i)
                        do i = 1, np
                            block
                                real(dp) :: val, lam
                                val = particles(i) + noise(i, local_t)
                                if (val < eps) val = eps
                                particles(i) = val
                                lam = val
                                if (lam < 1.0d-8) lam = 1.0d-8
                                loglik(i) = y(t_idx) * log(lam) - lam - log_gamma_y
                                if (loglik(i) > max_ll) max_ll = loglik(i)
                            end block
                        end do
                        !$omp end parallel do
                    end block
                end select

                ! ========================================
                ! 重み計算と正規化（1パス）
                ! ========================================
                w_sum = 0.0d0
                !$omp parallel do schedule(static) reduction(+:w_sum)
                do i = 1, np
                    weights(i) = exp(loglik(i) - max_ll)
                    w_sum = w_sum + weights(i)
                end do
                !$omp end parallel do

                if (w_sum <= 0.0d0) then
                    !$omp parallel do schedule(static)
                    do i = 1, np
                        weights(i) = 1.0d0 / np
                    end do
                    !$omp end parallel do
                    w_sum = 1.0d0
                end if

                ! 正規化と推定値計算を統合
                mean_val = 0.0d0
                !$omp parallel do schedule(static) reduction(+:mean_val)
                do i = 1, np
                    weights(i) = weights(i) / w_sum
                    mean_val = mean_val + particles(i) * weights(i)
                end do
                !$omp end parallel do
                xhat_out(t_idx) = mean_val

                ! ========================================
                ! 並列累積和 + リサンプリング
                ! ========================================
                call parallel_cumsum(weights, cumsum, np, block_sums, num_blocks, block_size)
                call parallel_systematic_resample(cumsum, np, indices)
                
                ! 粒子の再配置
                !$omp parallel do schedule(static)
                do i = 1, np
                    particles_new(i) = particles(indices(i))
                end do
                !$omp end parallel do
                
                call swap_alloc(particles, particles_new)
            end do
        end do

        deallocate(particles, loglik, weights, particles_new, indices, cumsum, block_sums, noise)
    end subroutine run_pf_fast
    
    !---------------------------------------------------------------------------
    ! チャンク単位ノイズ生成（並列）
    !---------------------------------------------------------------------------
    subroutine generate_noise_chunk(noise, np, n_steps, sigma)
        integer, intent(in) :: np, n_steps
        real(dp), intent(in) :: sigma
        real(dp), intent(out) :: noise(np, n_steps)
        integer :: t_idx, i, tid
        real(dp) :: z1, z2
        
        !$omp parallel private(tid, z1, z2)
        tid = omp_get_thread_num() + 1
        !$omp do schedule(static) collapse(2)
        do t_idx = 1, n_steps
            do i = 1, np, 2
                call xoshiro_randn_pair(tid, z1, z2)
                noise(i, t_idx) = z1 * sigma
                if (i + 1 <= np) noise(i+1, t_idx) = z2 * sigma
            end do
        end do
        !$omp end do
        !$omp end parallel
    end subroutine generate_noise_chunk

    !---------------------------------------------------------------------------
    ! 並列累積和（階層的アプローチ）
    !---------------------------------------------------------------------------
    subroutine parallel_cumsum(weights, cumsum, np, block_sums, num_blocks, block_size)
        integer, intent(in) :: np, num_blocks, block_size
        real(dp), intent(in) :: weights(np)
        real(dp), intent(out) :: cumsum(np)
        real(dp), intent(out) :: block_sums(num_blocks + 1)
        integer :: b, i, start_idx, end_idx
        real(dp) :: local_sum, prefix
        
        ! フェーズ1: 各ブロック内の累積和 + ブロック合計を同時計算
        !$omp parallel do schedule(static) private(start_idx, end_idx, local_sum, i)
        do b = 1, num_blocks
            start_idx = (b - 1) * block_size + 1
            end_idx = min(b * block_size, np)
            
            local_sum = 0.0d0
            do i = start_idx, end_idx
                local_sum = local_sum + weights(i)
                cumsum(i) = local_sum
            end do
            block_sums(b) = local_sum
        end do
        !$omp end parallel do
        
        ! フェーズ2: ブロック合計のプレフィックスサム（逐次、小さいので問題なし）
        prefix = 0.0d0
        do b = 1, num_blocks
            local_sum = block_sums(b)
            block_sums(b) = prefix
            prefix = prefix + local_sum
        end do
        
        ! フェーズ3: 各ブロックにオフセットを加算
        !$omp parallel do schedule(static) private(start_idx, end_idx, i)
        do b = 2, num_blocks
            start_idx = (b - 1) * block_size + 1
            end_idx = min(b * block_size, np)
            do i = start_idx, end_idx
                cumsum(i) = cumsum(i) + block_sums(b)
            end do
        end do
        !$omp end parallel do
    end subroutine parallel_cumsum

    !---------------------------------------------------------------------------
    ! 並列系統的リサンプリング
    !---------------------------------------------------------------------------
    subroutine parallel_systematic_resample(cumsum, np, indices)
        integer, intent(in) :: np
        real(dp), intent(in) :: cumsum(np)
        integer, intent(out) :: indices(np)
        real(dp) :: u_base, target
        integer :: j, low, high, mid, tid
        
        ! 一様乱数を1つ生成
        tid = 1
        !$omp parallel
            !$omp master
                u_base = xoshiro_uniform(1) / np
            !$omp end master
        !$omp end parallel
        
        ! 各粒子のインデックスを二分探索で並列に求める
        !$omp parallel do schedule(static) private(target, low, high, mid)
        do j = 1, np
            target = (j - 1.0d0) / np + u_base
            
            ! 二分探索
            low = 1
            high = np
            do while (low < high)
                mid = (low + high) / 2
                if (cumsum(mid) < target) then
                    low = mid + 1
                else
                    high = mid
                end if
            end do
            indices(j) = low
        end do
        !$omp end parallel do
    end subroutine parallel_systematic_resample

    !---------------------------------------------------------------------------
    ! ユーティリティ
    !---------------------------------------------------------------------------
    function calc_rmse(true_vals, est_vals, n) result(rmse)
        integer, intent(in) :: n
        real(dp), intent(in) :: true_vals(n), est_vals(n)
        real(dp) :: rmse, mse
        integer :: i
        mse = 0.0d0
        !$omp parallel do reduction(+:mse)
        do i = 1, n
            mse = mse + (true_vals(i) - est_vals(i))**2
        end do
        !$omp end parallel do
        rmse = sqrt(mse / n)
    end function calc_rmse

    subroutine calc_stats(times_arr, n, mean_val, std_val)
        integer, intent(in) :: n
        real(dp), intent(in) :: times_arr(n)
        real(dp), intent(out) :: mean_val, std_val
        real(dp) :: var
        integer :: i
        mean_val = sum(times_arr) / n
        var = 0.0d0
        do i = 1, n
            var = var + (times_arr(i) - mean_val)**2
        end do
        std_val = sqrt(var / n)
    end subroutine calc_stats
    
    subroutine swap_alloc(a, b)
        real(dp), allocatable, intent(inout) :: a(:), b(:)
        real(dp), allocatable :: tmp(:)
        call move_alloc(a, tmp)
        call move_alloc(b, a)
        call move_alloc(tmp, b)
    end subroutine swap_alloc

    subroutine load_data(filename, x1, y1, x2, y2, x3, y3, n)
        character(len=*), intent(in) :: filename
        integer, intent(in) :: n
        real(dp), intent(out) :: x1(n), y1(n), x2(n), y2(n), x3(n), y3(n)
        integer :: unit_num, i, t_val
        character(len=256) :: header
        
        open(newunit=unit_num, file=filename, status='old')
        read(unit_num, '(A)') header
        do i = 1, n
            read(unit_num, *) t_val, x1(i), y1(i), x2(i), y2(i), x3(i), y3(i)
        end do
        close(unit_num)
    end subroutine load_data

end program particle_filter_benchmark_fast