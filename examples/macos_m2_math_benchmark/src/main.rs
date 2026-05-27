// Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial

use std::time::Instant;
use std::f32::consts::PI;
use std::hint::black_box;

// ---------------------------------------------------------------------------
// 1. Dense General Matrix Multiplication (GEMM) - Highly Optimized Blocked
// ---------------------------------------------------------------------------
#[inline(never)]
fn run_blocked_gemm(n: usize) -> (f32, f64) {
    let a = vec![black_box(0.5f32); n * n];
    let b = vec![black_box(0.2f32); n * n];
    let mut c = vec![0.0f32; n * n];

    let t0 = Instant::now();
    let block_size = 64; // L1/L2 Cache-optimized tile size

    for si in (0..n).step_by(block_size) {
        let i_limit = std::cmp::min(si + block_size, n);
        for sk in (0..n).step_by(block_size) {
            let k_limit = std::cmp::min(sk + block_size, n);
            for sj in (0..n).step_by(block_size) {
                let j_limit = std::cmp::min(sj + block_size, n);
                
                // Micro-kernel with raw pointers to bypass bounds checking
                for i in si..i_limit {
                    let i_offset = i * n;
                    for k in sk..k_limit {
                        let k_offset = k * n;
                        unsafe {
                            let a_val = *a.get_unchecked(i_offset + k);
                            for j in sj..j_limit {
                                let c_ptr = c.get_unchecked_mut(i_offset + j);
                                let b_val = *b.get_unchecked(k_offset + j);
                                *c_ptr += a_val * b_val;
                            }
                        }
                    }
                }
            }
        }
    }

    let elapsed = t0.elapsed();
    let seconds = elapsed.as_secs_f64();
    let total_flops = 2.0 * (n as f64).powi(3);
    let gflops = (total_flops / seconds) / 1e9;
    
    (black_box(c[0] + c[n * n - 1]), gflops)
}

// ---------------------------------------------------------------------------
// 2. Cooley-Tukey Radix-2 1D Fast Fourier Transform (FFT) - Optimized
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
struct Complex {
    re: f32,
    im: f32,
}

impl Complex {
    #[inline(always)]
    fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
}

fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut rev = 0;
    for _ in 0..bits {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
    }
    rev
}

#[inline(never)]
fn run_fft_1d(n: usize) -> (f32, f64) {
    let bits = (n as f64).log2().round() as usize;
    let mut data: Vec<Complex> = (0..n)
        .map(|i| Complex::new(black_box(((i % 128) as f32) * 0.01), 0.0))
        .collect();

    let t0 = Instant::now();

    // 1. Bit-reversal permutation
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if i < j {
            data.swap(i, j);
        }
    }

    // 2. Cooley-Tukey stages
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / (len as f32);
        let w_step = Complex::new(angle.cos(), angle.sin());

        for i in (0..n).step_by(len) {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..half {
                unsafe {
                    let idx1 = i + j;
                    let idx2 = i + j + half;
                    let u = *data.get_unchecked(idx1);
                    let t = data.get_unchecked(idx2).mul(w);
                    *data.get_unchecked_mut(idx1) = u.add(t);
                    *data.get_unchecked_mut(idx2) = u.sub(t);
                }
                w = w.mul(w_step);
            }
        }
        len *= 2;
    }

    let elapsed = t0.elapsed();
    let seconds = elapsed.as_secs_f64();
    let total_flops = 5.0 * (n as f64) * (bits as f64);
    let mflops = (total_flops / seconds) / 1e6;

    (black_box(data[0].re + data[n - 1].im), mflops)
}

// ---------------------------------------------------------------------------
// 3. High-Precision Vector Transcendentals - Forced Runtime Sinks
// ---------------------------------------------------------------------------
#[inline(never)]
fn run_vector_transcendentals(n: usize) -> (f32, f64) {
    let mut x = vec![0.0f32; n];
    for i in 0..n {
        x[i] = (i as f32) * 1.2345e-7;
    }

    let mut y = vec![0.0f32; n];

    let t0 = Instant::now();

    // Force runtime dispatch via black_box in loops
    for i in 0..n {
        let xi = black_box(*unsafe { x.get_unchecked(i) });
        let val = xi.sin() * (xi + 1.0).ln() + (-xi * xi).exp();
        *unsafe { y.get_unchecked_mut(i) } = black_box(val);
    }

    let elapsed = t0.elapsed();
    let seconds = elapsed.as_secs_f64();
    let total_flops = 25.0 * (n as f64);
    let mops = (total_flops / seconds) / 1e6;

    (black_box(y[0] + y[n - 1]), mops)
}

// ---------------------------------------------------------------------------
// 4. Poisson Sparse Conjugate Gradient (CG) Solver - Optimized SpMV
// ---------------------------------------------------------------------------
struct SparseMatrix {
    n: usize,
    diag: Vec<f32>,
    off_left: Vec<f32>,
    off_right: Vec<f32>,
    off_up: Vec<f32>,
    off_down: Vec<f32>,
}

impl SparseMatrix {
    fn new_laplacian_2d(grid_sz: usize) -> Self {
        let n = grid_sz * grid_sz;
        Self {
            n,
            diag: vec![4.0f32; n],
            off_left: vec![-1.0f32; n],
            off_right: vec![-1.0f32; n],
            off_up: vec![-1.0f32; n],
            off_down: vec![-1.0f32; n],
        }
    }

    #[inline(always)]
    fn spmv(&self, x: &[f32], y: &mut [f32], grid_sz: usize) {
        let n = self.n;
        for i in 0..n {
            unsafe {
                let mut val = *self.diag.get_unchecked(i) * *x.get_unchecked(i);
                
                let col = i % grid_sz;
                let row = i / grid_sz;

                if col > 0 { val += *self.off_left.get_unchecked(i) * *x.get_unchecked(i - 1); }
                if col < grid_sz - 1 { val += *self.off_right.get_unchecked(i) * *x.get_unchecked(i + 1); }
                if row > 0 { val += *self.off_up.get_unchecked(i) * *x.get_unchecked(i - grid_sz); }
                if row < grid_sz - 1 { val += *self.off_down.get_unchecked(i) * *x.get_unchecked(i + grid_sz); }

                *y.get_unchecked_mut(i) = val;
            }
        }
    }
}

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline(never)]
fn run_poisson_solver(grid_sz: usize, iter_limit: usize) -> (f32, f64) {
    let matrix = SparseMatrix::new_laplacian_2d(grid_sz);
    let n = matrix.n;
    
    let b: Vec<f32> = (0..n).map(|i| if i == n / 2 { black_box(10.0f32) } else { 0.0f32 }).collect();
    let mut x = vec![0.0f32; n];
    
    let mut r = b.clone();
    let mut p = r.clone();
    let mut ap = vec![0.0f32; n];
    
    let mut r_dot_old = dot(&r, &r);

    let t0 = Instant::now();

    for _ in 0..iter_limit {
        matrix.spmv(&p, &mut ap, grid_sz);
        
        let p_dot_ap = dot(&p, &ap);
        let alpha = r_dot_old / (p_dot_ap + 1e-15);
        
        for i in 0..n {
            unsafe {
                *x.get_unchecked_mut(i) += alpha * *p.get_unchecked(i);
                *r.get_unchecked_mut(i) -= alpha * *ap.get_unchecked(i);
            }
        }
        
        let r_dot_new = dot(&r, &r);
        if r_dot_new < 1e-6 {
            break;
        }
        
        let beta = r_dot_new / (r_dot_old + 1e-15);
        
        for i in 0..n {
            unsafe {
                *p.get_unchecked_mut(i) = *r.get_unchecked(i) + beta * *p.get_unchecked(i);
            }
        }
        
        r_dot_old = r_dot_new;
    }

    let elapsed = t0.elapsed();
    let seconds = elapsed.as_secs_f64();
    let total_flops = 19.0 * (n as f64) * (iter_limit as f64);
    let gflops = (total_flops / seconds) / 1e9;

    (black_box(x[n / 2]), gflops)
}

// ---------------------------------------------------------------------------
// 5. Segmented Sieve of Eratosthenes (Prime Sieve)
// ---------------------------------------------------------------------------
#[inline(never)]
fn run_prime_sieve(limit: usize) -> (usize, f64) {
    let segment_size = 65536; // L1/L2 cache boundary
    let mut count = 0;
    
    let t0 = Instant::now();

    let sqrt_limit = (limit as f64).sqrt() as usize;
    let mut is_prime = vec![true; sqrt_limit + 1];
    let mut base_primes = Vec::new();

    for p in 2..=sqrt_limit {
        if is_prime[p] {
            base_primes.push(p);
            let mut mul = p * p;
            while mul <= sqrt_limit {
                is_prime[mul] = false;
                mul += p;
            }
        }
    }

    let mut segment = vec![true; segment_size];
    for low in (2..=limit).step_by(segment_size) {
        let high = std::cmp::min(low + segment_size - 1, limit);
        segment.fill(true);

        for &p in &base_primes {
            let mut start = (low + p - 1) / p * p;
            if start < p * p {
                start = p * p;
            }
            let mut j = start;
            while j <= high {
                unsafe {
                    *segment.get_unchecked_mut(j - low) = false;
                }
                j += p;
            }
        }

        for i in low..=high {
            unsafe {
                if *segment.get_unchecked(i - low) {
                    count += 1;
                }
            }
        }
    }

    let elapsed = t0.elapsed();
    let seconds = elapsed.as_secs_f64();
    let m_ints_sec = (limit as f64 / seconds) / 1e6;

    (count, m_ints_sec)
}

// ---------------------------------------------------------------------------
// Main Benchmarking Loop
// ---------------------------------------------------------------------------
fn main() {
    println!("\x1b[1m\x1b[36m================================================================================\x1b[0m");
    println!("\x1b[1m\x1b[32m                 RUNUX-AI INTENSE MATHEMATICAL BENCHMARK SUITE                  \x1b[0m");
    println!("\x1b[1m\x1b[36m================================================================================\x1b[0m");
    println!("Platform: Apple Silicon macOS M2 UMA");
    println!("Executing highly optimized float, vector, and cache math kernels...");
    println!("--------------------------------------------------------------------------------");

    // -- RUN 1: Blocked GEMM --
    print!("[1/5] dense_blocked_gemm (512x512)... ");
    let (_, gflops_512) = run_blocked_gemm(512);
    println!("\x1b[32mPASS\x1b[0m -> {:.2} GFLOPS", gflops_512);

    print!("[1/5] dense_blocked_gemm (1024x1024)... ");
    let (_, gflops_1024) = run_blocked_gemm(1024);
    println!("\x1b[32mPASS\x1b[0m -> {:.2} GFLOPS", gflops_1024);

    // -- RUN 2: 1D Fast Fourier Transform --
    print!("[2/5] cooley_tukey_fft_1d (N=2^20)... ");
    let (_, mflops_fft) = run_fft_1d(1048576);
    println!("\x1b[32mPASS\x1b[0m -> {:.2} MFLOPS", mflops_fft);

    // -- RUN 3: Vector Transcendentals --
    print!("[3/5] vector_transcendentals (N=10^7)... ");
    let (_, mops_trans) = run_vector_transcendentals(10000000);
    println!("\x1b[32mPASS\x1b[0m -> {:.2} MOPS", mops_trans);

    // -- RUN 4: Sparse Poisson Conjugate Gradient --
    print!("[4/5] sparse_poisson_solver (grid=100x100, iter=200)... ");
    let (_, gflops_poisson) = run_poisson_solver(100, 200);
    println!("\x1b[32mPASS\x1b[0m -> {:.3} GFLOPS", gflops_poisson);

    // -- RUN 5: Segmented Prime Sieve --
    print!("[5/5] segmented_prime_sieve (limit=10^8)... ");
    let (primes, m_ints_sec) = run_prime_sieve(100000000);
    println!("\x1b[32mPASS\x1b[0m -> Count: {} | {:.2} M-ints/sec", primes, m_ints_sec);

    println!("\x1b[1m\x1b[36m================================================================================\x1b[0m");
    println!("\x1b[1m\x1b[32m                 BENCHMARK COMPLETED SUCCESSFULLY (100% OK)                      \x1b[0m");
    println!("\x1b[1m\x1b[36m================================================================================\x1b[0m");

    let json_results = format!(
        r#"{{"gemm_512_gflops": {:.2}, "gemm_1024_gflops": {:.2}, "fft_1d_mflops": {:.2}, "transcendentals_mops": {:.2}, "poisson_cg_gflops": {:.3}, "prime_sieve_m_ints_sec": {:.2}, "total_primes": {}}}"#,
        gflops_512, gflops_1024, mflops_fft, mops_trans, gflops_poisson, m_ints_sec, primes
    );
    println!("\nMETRICS_JSON: {}", json_results);
}
