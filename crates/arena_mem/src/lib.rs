// Copyright (c) 2026 Xavier Callens / Socrate AI. All Rights Reserved.
// SPDX-License-Identifier: LicenseRef-RunuX-Commercial
// This file is proprietary and confidential. Unauthorized copying,
// distribution, or use is strictly prohibited.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
//! RunuX Arena Memory — Deterministic, zero-fragmentation allocator for LLM
//!
//! Traditional heap allocation (`Vec`, `Box`) introduces unpredictable latency
//! from fragmentation, `malloc` lock contention, and GC-like coalescing. For
//! real-time LLM inference on edge RISC-V hardware, we need:
//!
//! 1. **Deterministic allocation** — O(1) bump-pointer allocation
//! 2. **Zero fragmentation** — contiguous memory blocks, no holes
//! 3. **Cache-friendly layout** — sequential access patterns for RVV
//! 4. **Instant deallocation** — reset the arena between generations
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │  Arena (4GB backing store on AIBOX-K3)           │
//! │                                                  │
//! │  ┌─────────┬─────────┬─────────┬──────────────┐  │
//! │  │ Weights │ KV-Cache│ Scratch │  Free        │  │
//! │  │ (mmap)  │ (paged) │ (bump)  │              │  │
//! │  └─────────┴─────────┴─────────┴──────────────┘  │
//! │  ^                    ^                           │
//! │  weight_end           scratch_ptr (grows →)       │
//! └──────────────────────────────────────────────────┘
//! ```
//!
//! # Memory Zones
//!
//! - **Weight Zone**: Read-only, memory-mapped model weights (GGUF zero-copy)
//! - **KV-Cache Zone**: Paged blocks for attention cache, supports eviction
//! - **Scratch Zone**: Bump-allocated ephemeral buffers for per-token computation

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use core::cell::Cell;

// ---------------------------------------------------------------------------
// Arena Configuration
// ---------------------------------------------------------------------------

/// Memory zone identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryZone {
    /// Model weights (read-only after loading)
    Weights,
    /// KV-cache (paged, supports eviction)
    KvCache,
    /// Scratch space (bump-allocated, reset per token)
    Scratch,
}

/// Arena allocator configuration.
#[derive(Debug, Clone)]
pub struct ArenaConfig {
    /// Total arena size in bytes
    pub total_bytes: usize,
    /// Fraction allocated to weights (e.g., 0.6 = 60%)
    pub weight_fraction: f32,
    /// Fraction allocated to KV-cache (e.g., 0.3 = 30%)
    pub kv_cache_fraction: f32,
    /// Alignment requirement in bytes (must be power of 2)
    pub alignment: usize,
    /// KV-cache page size in bytes
    pub kv_page_size: usize,
}

impl ArenaConfig {
    /// Configuration for BPI-F3 (4GB RAM, budget ~3GB for AI).
    pub fn for_bpi_f3() -> Self {
        Self {
            total_bytes: 3_221_225_472, // 3GB
            weight_fraction: 0.65,
            kv_cache_fraction: 0.25,
            alignment: 64, // Cache line aligned
            kv_page_size: 4096,
        }
    }

    /// Configuration for AIBOX-K3 (32GB RAM, budget ~28GB for AI).
    pub fn for_aibox_k3() -> Self {
        Self {
            total_bytes: 30_064_771_072, // 28GB
            weight_fraction: 0.50,
            kv_cache_fraction: 0.40,
            alignment: 128, // 1024-bit VLEN alignment
            kv_page_size: 65536, // 64KB pages for K3
        }
    }

    fn weight_bytes(&self) -> usize {
        (self.total_bytes as f64 * self.weight_fraction as f64) as usize
    }

    fn kv_bytes(&self) -> usize {
        (self.total_bytes as f64 * self.kv_cache_fraction as f64) as usize
    }

    fn scratch_bytes(&self) -> usize {
        self.total_bytes - self.weight_bytes() - self.kv_bytes()
    }
}

// ---------------------------------------------------------------------------
// Bump Allocator (Scratch Zone)
// ---------------------------------------------------------------------------

/// A simple bump (arena) allocator for ephemeral inference buffers.
///
/// Allocates sequentially from a contiguous buffer. Deallocation is
/// all-at-once by resetting the bump pointer. This gives O(1) allocation
/// with zero fragmentation.
pub struct BumpAllocator {
    /// Backing memory
    buffer: Vec<u8>,
    /// Current allocation pointer offset
    offset: Cell<usize>,
    /// Total capacity
    capacity: usize,
    /// Number of allocations made since last reset
    alloc_count: Cell<u32>,
    /// Peak usage watermark
    peak_usage: Cell<usize>,
}

impl BumpAllocator {
    /// Create a new bump allocator with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity],
            offset: Cell::new(0),
            capacity,
            alloc_count: Cell::new(0),
            peak_usage: Cell::new(0),
        }
    }

    /// Allocate a slice of `count` f32 values, returning the offset.
    ///
    /// Returns `None` if the arena is exhausted.
    pub fn alloc_f32(&self, count: usize, alignment: usize) -> Option<usize> {
        let bytes_needed = count * 4;
        let current = self.offset.get();

        // Align up
        let aligned = (current + alignment - 1) & !(alignment - 1);
        let new_offset = aligned + bytes_needed;

        if new_offset > self.capacity {
            return None; // OOM
        }

        self.offset.set(new_offset);
        self.alloc_count.set(self.alloc_count.get() + 1);

        if new_offset > self.peak_usage.get() {
            self.peak_usage.set(new_offset);
        }

        Some(aligned)
    }

    /// Get a mutable slice into the arena at the given offset.
    ///
    /// # Safety
    /// Caller must ensure `offset` and `count` are within bounds and
    /// that no other references to the same region exist.
    pub unsafe fn get_slice_mut(&mut self, offset: usize, count: usize) -> &mut [f32] {
        let ptr = self.buffer.as_mut_ptr().add(offset) as *mut f32;
        core::slice::from_raw_parts_mut(ptr, count)
    }

    /// Get an immutable slice at the given offset.
    pub fn get_slice(&self, offset: usize, count: usize) -> &[f32] {
        unsafe {
            let ptr = self.buffer.as_ptr().add(offset) as *const f32;
            core::slice::from_raw_parts(ptr, count)
        }
    }

    /// Reset the allocator, freeing all allocations instantly.
    ///
    /// This is called between token generations to reclaim scratch space.
    pub fn reset(&self) {
        self.offset.set(0);
        self.alloc_count.set(0);
    }

    /// Current usage in bytes.
    pub fn usage_bytes(&self) -> usize {
        self.offset.get()
    }

    /// Peak usage watermark in bytes.
    pub fn peak_bytes(&self) -> usize {
        self.peak_usage.get()
    }

    /// Remaining capacity in bytes.
    pub fn remaining_bytes(&self) -> usize {
        self.capacity - self.offset.get()
    }

    /// Number of allocations since last reset.
    pub fn alloc_count(&self) -> u32 {
        self.alloc_count.get()
    }

    /// Utilization as a percentage.
    pub fn utilization_percent(&self) -> f32 {
        if self.capacity == 0 { return 0.0; }
        (self.offset.get() as f32 / self.capacity as f32) * 100.0
    }
}

// ---------------------------------------------------------------------------
// Paged KV-Cache Manager
// ---------------------------------------------------------------------------

/// A page in the paged KV-cache.
#[derive(Debug, Clone)]
pub struct KvPage {
    /// Page index
    pub page_id: u32,
    /// Sequence positions stored in this page
    pub start_pos: usize,
    /// Number of positions stored
    pub num_positions: usize,
    /// Whether this page is currently in use
    pub active: bool,
    /// Last access timestamp (for LRU eviction)
    pub last_access: u64,
    /// Memory offset in the KV zone
    pub mem_offset: usize,
}

/// Paged KV-cache manager with LRU eviction.
///
/// Inspired by PagedAttention (vLLM): manages KV-cache memory as fixed-size
/// pages rather than contiguous per-sequence blocks. This eliminates
/// internal fragmentation when sequences have different lengths.
pub struct PagedKvCache {
    /// All pages
    pages: Vec<KvPage>,
    /// Free page list (indices into `pages`)
    free_list: Vec<u32>,
    /// Page size in bytes
    page_size: usize,
    /// Total number of pages
    num_pages: usize,
    /// Current logical timestamp
    timestamp: u64,
}

impl PagedKvCache {
    /// Create a new paged KV-cache.
    pub fn new(total_bytes: usize, page_size: usize) -> Self {
        let num_pages = total_bytes / page_size;
        let pages: Vec<KvPage> = (0..num_pages)
            .map(|i| KvPage {
                page_id: i as u32,
                start_pos: 0,
                num_positions: 0,
                active: false,
                last_access: 0,
                mem_offset: i * page_size,
            })
            .collect();

        let free_list: Vec<u32> = (0..num_pages as u32).rev().collect();

        Self {
            pages,
            free_list,
            page_size,
            num_pages,
            timestamp: 0,
        }
    }

    /// Allocate a page for storing KV entries starting at `start_pos`.
    pub fn allocate_page(&mut self, start_pos: usize, num_positions: usize) -> Option<u32> {
        let page_id = self.free_list.pop()?;
        let page = &mut self.pages[page_id as usize];
        page.start_pos = start_pos;
        page.num_positions = num_positions;
        page.active = true;
        page.last_access = self.timestamp;
        self.timestamp += 1;
        Some(page_id)
    }

    /// Free a page, returning it to the free list.
    pub fn free_page(&mut self, page_id: u32) {
        if (page_id as usize) < self.pages.len() {
            self.pages[page_id as usize].active = false;
            self.free_list.push(page_id);
        }
    }

    /// Evict the least-recently-used page.
    pub fn evict_lru(&mut self) -> Option<u32> {
        let mut oldest_id = None;
        let mut oldest_time = u64::MAX;

        for page in &self.pages {
            if page.active && page.last_access < oldest_time {
                oldest_time = page.last_access;
                oldest_id = Some(page.page_id);
            }
        }

        if let Some(id) = oldest_id {
            self.free_page(id);
        }

        oldest_id
    }

    /// Touch a page (update last access time).
    pub fn touch(&mut self, page_id: u32) {
        if (page_id as usize) < self.pages.len() {
            self.pages[page_id as usize].last_access = self.timestamp;
            self.timestamp += 1;
        }
    }

    /// Number of free pages.
    pub fn free_pages(&self) -> usize {
        self.free_list.len()
    }

    /// Number of active pages.
    pub fn active_pages(&self) -> usize {
        self.num_pages - self.free_list.len()
    }

    /// Memory usage in bytes.
    pub fn active_bytes(&self) -> usize {
        self.active_pages() * self.page_size
    }

    /// Utilization as a percentage.
    pub fn utilization_percent(&self) -> f32 {
        if self.num_pages == 0 { return 0.0; }
        (self.active_pages() as f32 / self.num_pages as f32) * 100.0
    }
}

// ---------------------------------------------------------------------------
// Inference Memory Planner
// ---------------------------------------------------------------------------

/// Memory budget report for a given model + hardware combination.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Model weights (bytes)
    pub weights_bytes: usize,
    /// KV-cache (bytes) at target sequence length
    pub kv_cache_bytes: usize,
    /// Scratch space needed per token (bytes)
    pub scratch_per_token_bytes: usize,
    /// Total required (bytes)
    pub total_required_bytes: usize,
    /// Available on hardware (bytes)
    pub available_bytes: usize,
    /// Whether the model fits in memory
    pub fits: bool,
    /// Maximum sequence length that fits
    pub max_seq_len: usize,
    /// Recommended quantization to fit
    pub recommended_quant: Option<&'static str>,
}

/// Plan memory allocation for a model on specific hardware.
pub fn plan_memory(
    model_params: u64,       // number of parameters
    bits_per_param: u32,     // quantization bits (4, 8, 16, 32)
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    n_layers: usize,
    target_seq_len: usize,
    available_ram: usize,    // bytes
) -> MemoryBudget {
    // Weight memory: params × bits / 8
    let weights_bytes = (model_params as usize * bits_per_param as usize) / 8;

    // KV-cache: 2 (K+V) × n_layers × seq_len × n_kv_heads × head_dim × bytes_per_element
    // Using FP16 for KV cache (2 bytes)
    let kv_per_token = 2 * n_layers * n_kv_heads * head_dim * 2;
    let kv_cache_bytes = kv_per_token * target_seq_len;

    // Scratch: intermediate activations per token
    // hidden_dim buffers for norm, attention, FFN (roughly 4× hidden)
    let hidden_dim = n_heads * head_dim;
    let scratch_per_token = hidden_dim * 4 * 6; // 6 intermediate buffers

    let total = weights_bytes + kv_cache_bytes + scratch_per_token;

    // Calculate max seq len that fits
    let remaining_after_weights = if available_ram > weights_bytes + scratch_per_token {
        available_ram - weights_bytes - scratch_per_token
    } else {
        0
    };
    let max_seq_len = if kv_per_token > 0 {
        remaining_after_weights / kv_per_token
    } else {
        0
    };

    let recommended_quant = if !fits_in_memory(total, available_ram) {
        if fits_in_memory(recalc_weight_bytes(model_params, 4) + kv_cache_bytes + scratch_per_token, available_ram) {
            Some("Q4_K_M (4-bit)")
        } else if fits_in_memory(recalc_weight_bytes(model_params, 2) + kv_cache_bytes + scratch_per_token, available_ram) {
            Some("IQ2_XS (2-bit)")
        } else {
            Some("Model too large for this hardware")
        }
    } else {
        None
    };

    MemoryBudget {
        weights_bytes,
        kv_cache_bytes,
        scratch_per_token_bytes: scratch_per_token,
        total_required_bytes: total,
        available_bytes: available_ram,
        fits: fits_in_memory(total, available_ram),
        max_seq_len,
        recommended_quant,
    }
}

fn fits_in_memory(required: usize, available: usize) -> bool {
    // Leave 10% headroom for OS/runtime
    required < (available * 9 / 10)
}

fn recalc_weight_bytes(params: u64, bits: u32) -> usize {
    (params as usize * bits as usize) / 8
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bump_alloc_basic() {
        let arena = BumpAllocator::new(1024);
        let offset = arena.alloc_f32(10, 4);
        assert!(offset.is_some());
        assert_eq!(arena.alloc_count(), 1);
        assert!(arena.usage_bytes() >= 40); // 10 × 4 bytes
    }

    #[test]
    fn test_bump_alloc_oom() {
        let arena = BumpAllocator::new(32);
        let result = arena.alloc_f32(100, 4); // needs 400 bytes, only 32 available
        assert!(result.is_none());
    }

    #[test]
    fn test_bump_reset() {
        let arena = BumpAllocator::new(1024);
        arena.alloc_f32(10, 4);
        arena.alloc_f32(20, 4);
        assert_eq!(arena.alloc_count(), 2);
        assert!(arena.usage_bytes() > 0);

        arena.reset();
        assert_eq!(arena.alloc_count(), 0);
        assert_eq!(arena.usage_bytes(), 0);
        assert!(arena.peak_bytes() > 0); // Peak watermark preserved
    }

    #[test]
    fn test_paged_kv_alloc_free() {
        let mut cache = PagedKvCache::new(4096, 256); // 16 pages
        assert_eq!(cache.free_pages(), 16);

        let p1 = cache.allocate_page(0, 4);
        assert!(p1.is_some());
        assert_eq!(cache.active_pages(), 1);
        assert_eq!(cache.free_pages(), 15);

        cache.free_page(p1.unwrap());
        assert_eq!(cache.active_pages(), 0);
        assert_eq!(cache.free_pages(), 16);
    }

    #[test]
    fn test_paged_kv_lru_eviction() {
        let mut cache = PagedKvCache::new(512, 256); // 2 pages
        let p1 = cache.allocate_page(0, 4).unwrap();
        let _p2 = cache.allocate_page(4, 4).unwrap();

        // Both pages in use, no free pages
        assert_eq!(cache.free_pages(), 0);

        // Touch p1 to make it more recent
        cache.touch(p1);

        // Evict LRU (should evict p2)
        let evicted = cache.evict_lru();
        assert!(evicted.is_some());
        assert_eq!(cache.free_pages(), 1);
    }

    #[test]
    fn test_memory_plan_qwen_0_5b_bpi_f3() {
        let budget = plan_memory(
            500_000_000, // 0.5B params
            4,           // Q4_K_M
            64,          // head_dim
            14,          // n_heads
            2,           // n_kv_heads
            24,          // n_layers
            4096,        // target seq len
            4_294_967_296usize, // 4GB RAM
        );

        assert!(budget.fits, "Qwen 0.5B Q4 should fit on BPI-F3");
        assert!(budget.max_seq_len > 1000, "Should support >1K context");
    }

    #[test]
    fn test_memory_plan_deepseek_1_5b_bpi_f3() {
        let budget = plan_memory(
            1_500_000_000, // 1.5B params
            4,             // Q4_K_M
            128,           // head_dim
            12,            // n_heads
            2,             // n_kv_heads
            28,            // n_layers
            4096,          // target seq len
            4_294_967_296usize, // 4GB RAM
        );

        // 1.5B × 4 bits = 750MB weights + ~280MB KV = ~1GB — should fit
        assert!(budget.fits, "DeepSeek 1.5B Q4 should fit on BPI-F3");
    }

    #[test]
    fn test_arena_config_bpi() {
        let config = ArenaConfig::for_bpi_f3();
        assert!(config.weight_fraction + config.kv_cache_fraction < 1.0);
        assert!(config.scratch_bytes() > 0);
    }
}
