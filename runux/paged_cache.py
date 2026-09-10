# ==============================================================================
# RunuX AI Runtime — Paged KV-Cache Memory Allocator
#
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# Licensed under LicenseRef-RunuX-Commercial.
#
# Paged virtual memory manager for KV tensors, eliminating memory fragmentation
# during high-concurrency multi-turn LLM inference.
# ==============================================================================

import math
import torch
from typing import List, Dict, Optional, Tuple, Any

class PagedKVCache:
    """
    Paged KV-Cache Manager.
    Pre-allocates a contiguous pool of physical memory blocks on the GPU
    and assigns physical block indices dynamically to logical sequence positions.
    """
    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        num_heads: int = 8,
        head_dim: int = 64,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Pre-allocate physical KV buffer pool: [num_blocks, 2, num_heads, block_size, head_dim]
        # Dim 1: index 0 for K, index 1 for V
        self.pool = torch.zeros(
            (num_blocks, 2, num_heads, block_size, head_dim),
            dtype=dtype,
            device=device
        )

        self.free_blocks: List[int] = list(reversed(range(num_blocks)))
        self.block_tables: Dict[int, List[int]] = {}  # seq_id -> list of block_ids
        self.seq_lengths: Dict[int, int] = {}         # seq_id -> current token length

    def allocate_sequence(self, seq_id: int):
        """Initializes a new logical sequence."""
        if seq_id in self.block_tables:
            raise ValueError(f"Sequence ID {seq_id} already exists")
        self.block_tables[seq_id] = []
        self.seq_lengths[seq_id] = 0

    def free_sequence(self, seq_id: int):
        """Releases physical blocks allocated to a sequence back to the free pool."""
        if seq_id not in self.block_tables:
            return
        blocks = self.block_tables.pop(seq_id)
        self.seq_lengths.pop(seq_id, None)
        self.free_blocks.extend(blocks)

    def append_kv(self, seq_id: int, k_token: torch.Tensor, v_token: torch.Tensor):
        """
        Appends a single token's KV projection to the sequence's paged cache.
        k_token: [num_heads, head_dim]
        v_token: [num_heads, head_dim]
        """
        curr_len = self.seq_lengths[seq_id]
        block_offset = curr_len % self.block_size

        # If starting a new block, acquire from free pool
        if block_offset == 0:
            if not self.free_blocks:
                raise RuntimeError("PagedKVCache Out Of Memory: No free blocks remaining")
            block_id = self.free_blocks.pop()
            self.block_tables[seq_id].append(block_id)
        else:
            block_id = self.block_tables[seq_id][-1]

        # Write directly into physical block in GPU memory
        self.pool[block_id, 0, :, block_offset, :] = k_token
        self.pool[block_id, 1, :, block_offset, :] = v_token

        self.seq_lengths[seq_id] = curr_len + 1

    def get_sequence_kv(self, seq_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstructs the full KV sequence for attention computation.
        Returns: (K, V) each of shape [num_heads, total_len, head_dim]
        """
        blocks = self.block_tables[seq_id]
        total_len = self.seq_lengths[seq_id]
        if total_len == 0:
            empty = torch.empty(self.num_heads, 0, self.head_dim, dtype=self.dtype, device=self.device)
            return empty, empty

        k_chunks = []
        v_chunks = []
        tokens_remaining = total_len

        for b_id in blocks:
            n = min(self.block_size, tokens_remaining)
            k_chunks.append(self.pool[b_id, 0, :, :n, :])
            v_chunks.append(self.pool[b_id, 1, :, :n, :])
            tokens_remaining -= n

        k_out = torch.cat(k_chunks, dim=1)
        v_out = torch.cat(v_chunks, dim=1)
        return k_out, v_out

    def memory_stats(self) -> Dict[str, Any]:
        """Calculates current physical pool utilization and fragmentation metric."""
        total_slots = self.num_blocks * self.block_size
        used_blocks = self.num_blocks - len(self.free_blocks)
        used_tokens = sum(self.seq_lengths.values())

        # Internal fragmentation: allocated blocks with unused token slots
        allocated_capacity = used_blocks * self.block_size
        internal_frag = (allocated_capacity - used_tokens) / max(1, allocated_capacity)

        pool_mb = (self.pool.numel() * self.pool.element_size()) / (1024**2)

        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "used_blocks": used_blocks,
            "free_blocks": len(self.free_blocks),
            "total_tokens_stored": used_tokens,
            "pool_vram_mb": round(pool_mb, 2),
            "internal_fragmentation_pct": round(internal_frag * 100.0, 2),
            "external_fragmentation_pct": 0.0  # Paged memory has 0% external fragmentation by design
        }
