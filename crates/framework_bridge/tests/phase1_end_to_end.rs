// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial

//! Phase 1 End-to-End Integration Test
//!
//! Validates the joint interaction of:
//! - Memory Subsystem: `arena_mem` (BumpAllocator, PagedKvCache)
//! - Attention Engine: `flash_attention` (Tiled FlashAttention-2)
//! - Compression Engine: `turbo_quant` (PolarQuant orthogonal rotation + compressed cache)
//! - GPU Acceleration: `gpu_compute` (GpuContext embedding lookup + matmul)
//! - Bridge Layer: `framework_bridge` (C FFI for PyTorch/JAX)

use arena_mem::{ArenaConfig, BumpAllocator, PagedKvCache};
use flash_attention::{flash_attention_forward, FlashAttentionConfig};
use framework_bridge::{
    runux_flash_attention, runux_matmul, runux_rms_norm, runux_silu,
};
use gpu_compute::GpuContext;
use turbo_quant::{CompressedKvCache, PolarQuant, TurboQuantConfig};

#[test]
fn test_phase1_full_pipeline_integration() {
    // -----------------------------------------------------------------------
    // Step 1: Memory Arena Allocation (Arena-Mem)
    // -----------------------------------------------------------------------
    let _arena_config = ArenaConfig::for_aibox_k3();
    let mut arena = BumpAllocator::new(1024 * 1024); // 1 MB scratch
    assert_eq!(arena.remaining_bytes(), 1024 * 1024);

    let batch = 1;
    let seq_len = 16;
    let heads = 2;
    let head_dim = 32;
    let total_elements = batch * seq_len * heads * head_dim;

    let q_offset = arena.alloc_f32(total_elements, 64).expect("Alloc Q");
    let k_offset = arena.alloc_f32(total_elements, 64).expect("Alloc K");
    let v_offset = arena.alloc_f32(total_elements, 64).expect("Alloc V");
    let out_offset = arena.alloc_f32(total_elements, 64).expect("Alloc Out");

    // Populate input tensors with deterministic data
    unsafe {
        let q_slice = arena.get_slice_mut(q_offset, total_elements);
        for (i, val) in q_slice.iter_mut().enumerate() {
            *val = ((i % 17) as f32) * 0.05;
        }

        let k_slice = arena.get_slice_mut(k_offset, total_elements);
        for (i, val) in k_slice.iter_mut().enumerate() {
            *val = ((i % 13) as f32) * 0.04;
        }

        let v_slice = arena.get_slice_mut(v_offset, total_elements);
        for (i, val) in v_slice.iter_mut().enumerate() {
            *val = ((i % 11) as f32) * 0.03;
        }
    }

    // -----------------------------------------------------------------------
    // Step 2: FlashAttention Execution (Kernel + FFI Bridge)
    // -----------------------------------------------------------------------
    // Direct kernel
    let flash_cfg = FlashAttentionConfig {
        n_heads: heads,
        head_dim,
        tile_q: 16,
        tile_kv: 16,
        causal: true,
        scale: 1.0 / (head_dim as f32).sqrt(),
    };
    let kernel_out = flash_attention_forward(
        arena.get_slice(q_offset, total_elements),
        arena.get_slice(k_offset, total_elements),
        arena.get_slice(v_offset, total_elements),
        &flash_cfg,
    );
    assert_eq!(kernel_out.output.len(), total_elements);
    assert!(kernel_out.flops > 0);

    // C FFI Bridge (PyTorch/JAX entry point)
    let q_ptr = arena.get_slice(q_offset, total_elements).as_ptr();
    let k_ptr = arena.get_slice(k_offset, total_elements).as_ptr();
    let v_ptr = arena.get_slice(v_offset, total_elements).as_ptr();
    let out_ptr = unsafe { arena.get_slice_mut(out_offset, total_elements).as_mut_ptr() };

    let ffi_res = unsafe {
        runux_flash_attention(
            q_ptr,
            k_ptr,
            v_ptr,
            out_ptr,
            batch as i32,
            heads as i32,
            seq_len as i32,
            head_dim as i32,
            1, // causal = true
            3, // backend = GPU (CUDA/T4 simulation fallback)
        )
    };
    assert_eq!(ffi_res, 0, "FFI call to runux_flash_attention must succeed");

    // Verify FFI output is finite
    let ffi_out = arena.get_slice(out_offset, total_elements);
    assert!(ffi_out.iter().all(|v| v.is_finite()));

    // -----------------------------------------------------------------------
    // Step 3: Paged KV Cache Management (PagedKvCache)
    // -----------------------------------------------------------------------
    let mut paged_cache = PagedKvCache::new(4096, 64);
    let page0 = paged_cache.allocate_page(0, 16).expect("Allocate page 0");
    let _page1 = paged_cache.allocate_page(16, 16).expect("Allocate page 1");
    assert_eq!(paged_cache.active_pages(), 2);

    let test_kv_data = [42u8; 32];
    paged_cache.write_page_data(page0, 0, &test_kv_data).expect("Write page 0");

    let readback = paged_cache.read_page_data(page0, 0, 32).expect("Read page 0");
    assert_eq!(readback[0], 42u8);

    // -----------------------------------------------------------------------
    // Step 4: TurboQuant Compression & Polar Quantization (Turbo-Quant)
    // -----------------------------------------------------------------------
    let polar = PolarQuant::new(42, head_dim);
    let sample_vec = &ffi_out[0..head_dim];
    let mut rotated = vec![0.0f32; head_dim];
    let mut reconstructed = vec![0.0f32; head_dim];

    polar.rotate_forward(sample_vec, &mut rotated);
    polar.rotate_inverse(&rotated, &mut reconstructed);

    let norm_orig: f32 = sample_vec.iter().map(|x| x * x).sum();
    let norm_rot: f32 = rotated.iter().map(|x| x * x).sum();
    if norm_orig > 1e-6 {
        let diff = (norm_orig - norm_rot).abs() / norm_orig;
        assert!(diff < 1e-3, "PolarQuant must preserve vector norm");
    }

    let tq_cfg = TurboQuantConfig::default();
    let compressed_cache = CompressedKvCache::new(tq_cfg, 1, head_dim, heads);
    assert_eq!(compressed_cache.memory_bytes(), 0);

    // -----------------------------------------------------------------------
    // Step 5: GPU Compute & Embedding Lookup (GpuContext)
    // -----------------------------------------------------------------------
    let gpu = GpuContext::new().expect("Create GpuContext");
    assert!(gpu.is_ready());

    let vocab_size = 100;
    let embed_table: Vec<f32> = (0..vocab_size * head_dim)
        .map(|i| (i as f32) * 0.001)
        .collect();
    let tokens = [3, 7, 12, 45];
    let mut embed_out = vec![0.0f32; tokens.len() * head_dim];
    gpu.embedding_lookup(&tokens, &embed_table, &mut embed_out)
        .expect("Embedding lookup");
    assert_eq!(embed_out.len(), tokens.len() * head_dim);

    // -----------------------------------------------------------------------
    // Step 6: Post-Attention Activations via Framework Bridge (FFI)
    // -----------------------------------------------------------------------
    let mut act_buf = embed_out.clone();
    let norm_weight = vec![1.0f32; act_buf.len()];
    let status_norm = unsafe {
        runux_rms_norm(
            act_buf.as_mut_ptr(),
            norm_weight.as_ptr(),
            act_buf.len() as i32,
            1e-5,
        )
    };
    assert_eq!(status_norm, 0);

    let status_silu = unsafe {
        runux_silu(act_buf.as_mut_ptr(), act_buf.len() as i32)
    };
    assert_eq!(status_silu, 0);

    // -----------------------------------------------------------------------
    // Step 7: Linear Projection MatMul (FFI)
    // -----------------------------------------------------------------------
    let m = tokens.len();
    let k = head_dim;
    let n = 16;
    let weights = vec![0.01f32; k * n];
    let mut proj_out = vec![0.0f32; m * n];

    let status_matmul = unsafe {
        runux_matmul(
            act_buf.as_ptr(),
            weights.as_ptr(),
            proj_out.as_mut_ptr(),
            m as i32,
            n as i32,
            k as i32,
            3, // GPU backend
        )
    };
    assert_eq!(status_matmul, 0);
    assert!(proj_out.iter().all(|v| v.is_finite()));
}
