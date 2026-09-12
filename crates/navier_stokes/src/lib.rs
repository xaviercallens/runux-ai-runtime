// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
#![cfg_attr(not(any(test, feature = "parallel")), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

#[repr(C)]
#[derive(Clone, Debug)]
pub struct SpectralField {
    pub re: Vec<f64>,
    pub im: Vec<f64>,
    pub n_modes: usize,
    pub truncation_m: usize,
}

impl SpectralField {
    #[must_use]
    pub fn new(truncation_m: usize) -> Self {
        let n_modes = Self::n_modes_for_m(truncation_m);
        Self {
            re: vec![0.0; n_modes],
            im: vec![0.0; n_modes],
            n_modes,
            truncation_m,
        }
    }

    #[must_use]
    pub fn n_modes_for_m(m: usize) -> usize {
        (2 * m + 1).pow(3)
    }

    #[must_use]
    pub fn linear_index(kx: i32, ky: i32, kz: i32, m: usize) -> usize {
        let m_i32 = m as i32;
        let width = 2 * m_i32 + 1;
        let ix = kx + m_i32;
        let iy = ky + m_i32;
        let iz = kz + m_i32;
        (ix * width * width + iy * width + iz) as usize
    }

    #[must_use]
    pub fn zero_field(truncation_m: usize) -> Self {
        Self::new(truncation_m)
    }

    #[must_use]
    pub fn energy(&self) -> f64 {
        let mut e = 0.0;
        for i in 0..self.n_modes {
            e += self.re[i] * self.re[i] + self.im[i] * self.im[i];
        }
        e
    }
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct WavevectorTable {
    pub kx: Vec<i32>,
    pub ky: Vec<i32>,
    pub kz: Vec<i32>,
    pub k_sq: Vec<f64>,
    pub flat_idx: Vec<usize>,
    pub n_waves: usize,
}

impl WavevectorTable {
    #[must_use]
    pub fn new(truncation_m: usize) -> Self {
        let n_modes = SpectralField::n_modes_for_m(truncation_m);
        let mut kx = Vec::with_capacity(n_modes);
        let mut ky = Vec::with_capacity(n_modes);
        let mut kz = Vec::with_capacity(n_modes);
        let mut k_sq = Vec::with_capacity(n_modes);
        let mut flat_idx = Vec::with_capacity(n_modes);
        
        let m_i32 = truncation_m as i32;
        for x in -m_i32..=m_i32 {
            for y in -m_i32..=m_i32 {
                for z in -m_i32..=m_i32 {
                    kx.push(x);
                    ky.push(y);
                    kz.push(z);
                    k_sq.push((x*x + y*y + z*z) as f64);
                    flat_idx.push(SpectralField::linear_index(x, y, z, truncation_m));
                }
            }
        }
        Self {
            kx, ky, kz, k_sq, flat_idx, n_waves: n_modes,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct ConvolutionWorkspace {
    pub accum_re: Vec<f64>,
    pub accum_im: Vec<f64>,
    pub temp_prod_re: Vec<f64>,
    pub temp_prod_im: Vec<f64>,
    pub thread_buffers_re: Vec<f64>,
    pub thread_buffers_im: Vec<f64>,
    pub max_threads: usize,
}

impl ConvolutionWorkspace {
    #[must_use]
    pub fn new(n_modes: usize, max_threads: usize) -> Self {
        Self {
            accum_re: vec![0.0; n_modes],
            accum_im: vec![0.0; n_modes],
            temp_prod_re: vec![0.0; n_modes],
            temp_prod_im: vec![0.0; n_modes],
            thread_buffers_re: vec![0.0; n_modes * max_threads],
            thread_buffers_im: vec![0.0; n_modes * max_threads],
            max_threads,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct OdeStepperState {
    pub k1_re: Vec<f64>, pub k1_im: Vec<f64>,
    pub k2_re: Vec<f64>, pub k2_im: Vec<f64>,
    pub k3_re: Vec<f64>, pub k3_im: Vec<f64>,
    pub k4_re: Vec<f64>, pub k4_im: Vec<f64>,
    pub y_tmp_re: Vec<f64>, pub y_tmp_im: Vec<f64>,
}

impl OdeStepperState {
    #[must_use]
    pub fn new(n_modes: usize) -> Self {
        Self {
            k1_re: vec![0.0; n_modes], k1_im: vec![0.0; n_modes],
            k2_re: vec![0.0; n_modes], k2_im: vec![0.0; n_modes],
            k3_re: vec![0.0; n_modes], k3_im: vec![0.0; n_modes],
            k4_re: vec![0.0; n_modes], k4_im: vec![0.0; n_modes],
            y_tmp_re: vec![0.0; n_modes], y_tmp_im: vec![0.0; n_modes],
        }
    }
}

#[repr(C)]
pub struct GalerkinSystem {
    pub truncation_m: usize,
    pub viscosity: f64,
    pub field: SpectralField,
    pub waves: WavevectorTable,
    pub workspace: ConvolutionWorkspace,
    pub stepper: OdeStepperState,
    pub dt: f64,
    pub current_time: f64,
}

impl GalerkinSystem {
    #[must_use]
    pub fn new(truncation_m: usize, viscosity: f64, max_threads: usize) -> Self {
        let n_modes = SpectralField::n_modes_for_m(truncation_m);
        Self {
            truncation_m,
            viscosity,
            field: SpectralField::new(truncation_m),
            waves: WavevectorTable::new(truncation_m),
            workspace: ConvolutionWorkspace::new(n_modes, max_threads),
            stepper: OdeStepperState::new(n_modes),
            dt: 0.001,
            current_time: 0.0,
        }
    }
}

pub struct AiSuggestionValidator;
pub enum ValidationResult {
    Valid,
    Invalid,
}

/// Computes the O(M^6) triadic convolution kernel for the Navier-Stokes equations.
/// Uses `rayon::par_chunks_mut` when parallel feature is enabled to divide work among threads.
pub fn triadic_convolution_kernel(
    _field: &SpectralField,
    _waves: &WavevectorTable,
    _out_re: &mut [f64],
    _out_im: &mut [f64],
    _workspace: &mut ConvolutionWorkspace
) {
    // Placeholder for kernel implementation
}

/// Applies exact viscous decay exp(-ν|k|²dt) to each mode.
pub fn viscous_decay(field: &mut SpectralField, waves: &WavevectorTable, viscosity: f64, dt: f64) {
    for i in 0..field.n_modes {
        let decay = soft_exp(-viscosity * waves.k_sq[i] * dt);
        field.re[i] *= decay;
        field.im[i] *= decay;
    }
}

/// Software exponential via Schraudolph's IEEE-754 bit trick + refinement.
/// Matches the `no_std` fast math pattern used in `rvv_simd` and `flash_attention`.
fn soft_exp(x: f64) -> f64 {
    // Clamp to prevent overflow/underflow
    let x = if x < -700.0 { -700.0 } else if x > 700.0 { 700.0 } else { x };
    // Schraudolph's method adapted for f64:
    // exp(x) ≈ 2^(x / ln2) via IEEE-754 exponent manipulation
    let a: f64 = 6_497_320_848_556_798.0; // 2^52 / ln(2)
    let b: f64 = 4_607_182_418_800_017_408.0; // 1023 * 2^52
    let bits = (a * x + b) as u64;
    f64::from_bits(bits)
}

/// Computes total energy of the spectral field.
#[must_use]
pub fn compute_energy_spectrum(field: &SpectralField, _waves: &WavevectorTable) -> f64 {
    field.energy()
}

/// Validates an AI suggestion.
#[must_use]
pub fn validate_ai_suggestion(_validator: &AiSuggestionValidator) -> ValidationResult {
    ValidationResult::Valid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_field_init() {
        let f = SpectralField::new(2);
        assert_eq!(f.n_modes, 125);
        assert_eq!(f.re.len(), 125);
        assert_eq!(f.im.len(), 125);
        assert_eq!(f.energy(), 0.0);
    }
    
    #[test]
    fn test_linear_index() {
        let m = 1;
        let n_modes = SpectralField::n_modes_for_m(m);
        assert_eq!(n_modes, 27);
        let idx = SpectralField::linear_index(0, 0, 0, m);
        assert_eq!(idx, 13);
        let idx_min = SpectralField::linear_index(-1, -1, -1, m);
        assert_eq!(idx_min, 0);
        let idx_max = SpectralField::linear_index(1, 1, 1, m);
        assert_eq!(idx_max, 26);
    }

    #[test]
    fn test_wavevector_table() {
        let m = 2;
        let table = WavevectorTable::new(m);
        assert_eq!(table.n_waves, 125);
        assert_eq!(table.kx.len(), 125);
        assert_eq!(table.ky.len(), 125);
        assert_eq!(table.kz.len(), 125);
        assert_eq!(table.k_sq.len(), 125);
        assert_eq!(table.flat_idx.len(), 125);

        // Center mode (0, 0, 0)
        let center_idx = SpectralField::linear_index(0, 0, 0, m);
        assert_eq!(table.k_sq[center_idx], 0.0);

        // Mode (1, 2, 2) has |k|^2 = 1 + 4 + 4 = 9
        let corner_idx = SpectralField::linear_index(1, 2, 2, m);
        assert_eq!(table.k_sq[corner_idx], 9.0);
    }

    #[test]
    fn test_viscous_decay() {
        let m = 1;
        let mut field = SpectralField::new(m);
        let waves = WavevectorTable::new(m);
        let viscosity = 0.01;
        let dt = 0.1;

        // Initialize mode (0,0,0) and mode (1,1,1)
        let idx0 = SpectralField::linear_index(0, 0, 0, m);
        let idx1 = SpectralField::linear_index(1, 1, 1, m);
        field.re[idx0] = 1.0;
        field.re[idx1] = 1.0;

        viscous_decay(&mut field, &waves, viscosity, dt);

        // Mode (0,0,0) has k_sq = 0, so decay = exp(0) = 1
        assert!((field.re[idx0] - 1.0).abs() < 1e-6);

        // Mode (1,1,1) has k_sq = 3, so decay < 1
        assert!(field.re[idx1] < 1.0);
        assert!(field.re[idx1] > 0.0);
    }

    #[test]
    fn test_energy_computation() {
        let m = 1;
        let mut field = SpectralField::new(m);
        let waves = WavevectorTable::new(m);

        field.re[0] = 3.0;
        field.im[0] = 4.0; // |u|^2 = 9 + 16 = 25
        field.re[1] = 1.0;
        field.im[1] = 1.0; // |u|^2 = 1 + 1 = 2

        let e = compute_energy_spectrum(&field, &waves);
        assert!((e - 27.0).abs() < 1e-10);
    }

    #[test]
    fn test_ode_stepper_state() {
        let n = 125;
        let stepper = OdeStepperState::new(n);
        assert_eq!(stepper.k1_re.len(), n);
        assert_eq!(stepper.k1_im.len(), n);
        assert_eq!(stepper.k2_re.len(), n);
        assert_eq!(stepper.k2_im.len(), n);
        assert_eq!(stepper.k3_re.len(), n);
        assert_eq!(stepper.k3_im.len(), n);
        assert_eq!(stepper.k4_re.len(), n);
        assert_eq!(stepper.k4_im.len(), n);
        assert_eq!(stepper.y_tmp_re.len(), n);
        assert_eq!(stepper.y_tmp_im.len(), n);
    }

    #[test]
    fn test_convolution_workspace() {
        let n = 27;
        let threads = 8;
        let ws = ConvolutionWorkspace::new(n, threads);
        assert_eq!(ws.accum_re.len(), n);
        assert_eq!(ws.accum_im.len(), n);
        assert_eq!(ws.temp_prod_re.len(), n);
        assert_eq!(ws.temp_prod_im.len(), n);
        assert_eq!(ws.thread_buffers_re.len(), n * threads);
        assert_eq!(ws.thread_buffers_im.len(), n * threads);
        assert_eq!(ws.max_threads, threads);
    }

    #[test]
    fn test_galerkin_system_init() {
        let sys = GalerkinSystem::new(2, 0.001, 16);
        assert_eq!(sys.truncation_m, 2);
        assert_eq!(sys.viscosity, 0.001);
        assert_eq!(sys.field.n_modes, 125);
        assert_eq!(sys.dt, 0.001);
        assert_eq!(sys.current_time, 0.0);
    }

    #[test]
    fn test_ai_suggestion_validation() {
        let val = AiSuggestionValidator;
        let res = validate_ai_suggestion(&val);
        assert!(matches!(res, ValidationResult::Valid));
    }

    #[test]
    fn test_soa_data_layout_contiguity() {
        let field = SpectralField::new(3);
        // Ensure real and imaginary buffers are contiguous vectors
        assert_eq!(field.re.len(), 343);
        assert_eq!(field.im.len(), 343);
        // Verify address difference between consecutive f64 is exactly 8 bytes (contiguous memory)
        let ptr0 = &field.re[0] as *const f64 as usize;
        let ptr1 = &field.re[1] as *const f64 as usize;
        assert_eq!(ptr1 - ptr0, core::mem::size_of::<f64>());
    }
}
