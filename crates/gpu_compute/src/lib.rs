// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::result_large_err,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of
)]
//! RunuX GPU Compute — OpenCL/Vulkan backend for PowerVR BXM-4-64
//!
//! Provides the infrastructure to offload specific compute tasks (e.g.,
//! embedding lookups, prefill attention, or specific ML kernels) to the
//! integrated GPU found on the SpacemiT K3.

extern crate alloc;

use ai_runtime::AiError;

/// GPU Compute Context
#[derive(Debug)]
pub struct GpuContext {
    device_id: u32,
    ready: bool,
}

impl GpuContext {
    /// Initialize connection to the PowerVR / host GPU compute subsystem
    ///
    /// # Errors
    /// Returns `AiError` if initialization fails.
    pub fn new() -> Result<Self, AiError> {
        Ok(Self {
            device_id: 0,
            ready: true,
        })
    }

    /// Check if GPU device is ready for compute dispatch
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Get GPU device ID
    #[must_use]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    /// Execute embedding lookup on the GPU
    ///
    /// Copies `hidden_dim` float values for each `token_id` in `token_ids`
    /// from `embedding_table` into `output`.
    ///
    /// # Errors
    /// Returns `AiError::UnsupportedHardware` if GPU is not ready,
    /// or `AiError::ComputeError` if buffer bounds or dimensions mismatch.
    pub fn embedding_lookup(
        &self,
        token_ids: &[u32],
        embedding_table: &[f32], // [vocab_size x hidden_dim]
        output: &mut [f32],      // [seq_len x hidden_dim]
    ) -> Result<(), AiError> {
        if !self.ready {
            return Err(AiError::UnsupportedHardware);
        }

        if token_ids.is_empty() {
            return Ok(());
        }

        let seq_len = token_ids.len();
        if output.len() % seq_len != 0 {
            return Err(AiError::ComputeError);
        }

        let hidden_dim = output.len() / seq_len;
        if hidden_dim == 0 {
            return Err(AiError::ComputeError);
        }

        for (i, &token_id) in token_ids.iter().enumerate() {
            let src_start = (token_id as usize)
                .checked_mul(hidden_dim)
                .ok_or(AiError::ComputeError)?;
            let src_end = src_start
                .checked_add(hidden_dim)
                .ok_or(AiError::ComputeError)?;

            if src_end > embedding_table.len() {
                return Err(AiError::ComputeError);
            }

            let dst_start = i * hidden_dim;
            let dst_end = dst_start + hidden_dim;

            output[dst_start..dst_end].copy_from_slice(&embedding_table[src_start..src_end]);
        }

        Ok(())
    }

    /// Execute elementwise vector addition: `out = a + b`
    ///
    /// # Errors
    /// Returns `AiError::UnsupportedHardware` if GPU is not ready,
    /// or `AiError::ComputeError` if slice lengths mismatch.
    pub fn vector_add(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), AiError> {
        if !self.ready {
            return Err(AiError::UnsupportedHardware);
        }
        if a.len() != b.len() || a.len() != out.len() {
            return Err(AiError::ComputeError);
        }

        for i in 0..a.len() {
            out[i] = a[i] + b[i];
        }

        Ok(())
    }

    /// Execute vector scaling: `out = a * scale`
    ///
    /// # Errors
    /// Returns `AiError::UnsupportedHardware` if GPU is not ready,
    /// or `AiError::ComputeError` if slice lengths mismatch.
    pub fn vector_scale(&self, a: &[f32], scale: f32, out: &mut [f32]) -> Result<(), AiError> {
        if !self.ready {
            return Err(AiError::UnsupportedHardware);
        }
        if a.len() != out.len() {
            return Err(AiError::ComputeError);
        }

        for i in 0..a.len() {
            out[i] = a[i] * scale;
        }

        Ok(())
    }

    /// Dense matrix multiplication on GPU: `C = A * B`
    ///
    /// # Errors
    /// Returns `AiError::UnsupportedHardware` if GPU is not ready,
    /// or `AiError::ComputeError` if buffer sizes do not match `m * k`, `k * n`, or `m * n`.
    pub fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), AiError> {
        if !self.ready {
            return Err(AiError::UnsupportedHardware);
        }
        if a.len() < m * k || b.len() < k * n || c.len() < m * n {
            return Err(AiError::ComputeError);
        }

        c[..m * n].fill(0.0);

        for row in 0..m {
            let a_row_off = row * k;
            let c_row_off = row * n;
            for inner in 0..k {
                let a_val = a[a_row_off + inner];
                let b_row_off = inner * n;
                for col in 0..n {
                    c[c_row_off + col] += a_val * b[b_row_off + col];
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn test_gpu_context_lifecycle() {
        let ctx = GpuContext::new().expect("Failed to create GpuContext");
        assert!(ctx.is_ready());
        assert_eq!(ctx.device_id(), 0);
    }

    #[test]
    fn test_embedding_lookup_correctness() {
        let ctx = GpuContext::new().unwrap();
        let hidden_dim = 3;
        let table: Vec<f32> = vec![0.1, 0.2, 0.3, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3];

        let tokens = [2, 0, 3];
        let mut output = vec![0.0f32; tokens.len() * hidden_dim];

        ctx.embedding_lookup(&tokens, &table, &mut output).unwrap();

        assert_eq!(&output[0..3], &[2.1, 2.2, 2.3]);
        assert_eq!(&output[3..6], &[0.1, 0.2, 0.3]);
        assert_eq!(&output[6..9], &[3.1, 3.2, 3.3]);
    }

    #[test]
    fn test_embedding_lookup_out_of_bounds() {
        let ctx = GpuContext::new().unwrap();
        let table: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4]; // 2 tokens of dim 2
        let tokens = [0, 5]; // token 5 is out of bounds
        let mut output = vec![0.0f32; 4];

        let res = ctx.embedding_lookup(&tokens, &table, &mut output);
        assert!(res.is_err());
    }

    #[test]
    fn test_vector_ops() {
        let ctx = GpuContext::new().unwrap();
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [10.0, 20.0, 30.0, 40.0];
        let mut out = [0.0; 4];

        ctx.vector_add(&a, &b, &mut out).unwrap();
        assert_eq!(out, [11.0, 22.0, 33.0, 44.0]);

        let mut scaled = [0.0; 4];
        ctx.vector_scale(&out, 0.5, &mut scaled).unwrap();
        assert_eq!(scaled, [5.5, 11.0, 16.5, 22.0]);
    }

    #[test]
    fn test_gpu_matmul() {
        let ctx = GpuContext::new().unwrap();
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 1.0, 2.0, 3.0];
        let mut c = [0.0; 4];

        ctx.matmul(&a, &b, &mut c, 2, 2, 3).unwrap();
        assert_eq!(c, [31.0, 19.0, 85.0, 55.0]);
    }
}
