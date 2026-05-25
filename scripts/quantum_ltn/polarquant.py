# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
#
# WARS-Quantum-LTN: PolarQuant 3-bit Matrix Compression Module
# ============================================================

import numpy as np
from typing import Tuple

class PolarQuantCompressor:
    """Simulates 3-bit PolarQuant boundary matrix compression for PEPS tensor contractions."""
    def __init__(self, target_bits: int = 3):
        self.target_bits = target_bits
        self.num_levels = 2 ** target_bits
        # Uniform levels in [-1.0, 1.0] for rotated boundary elements
        self.codebook = np.linspace(-1.0, 1.0, self.num_levels)

    def compress_matrix(self, matrix: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Compresses a boundary matrix using random orthogonal rotation and 3-bit quantization.
        """
        n = matrix.shape[0]
        original_memory = matrix.nbytes
        
        # 1. Generate pseudo-random orthogonal rotation matrix R
        H = np.random.normal(0.0, 1.0, (n, n))
        Q, R = np.linalg.qr(H)  # QR decomposition yields orthogonal Q
        
        # 2. PolarQuant Rotation (eliminates extreme outliers, preserves norm)
        rotated = np.dot(matrix, Q)
        
        # 3. 3-bit Uniform Quantization
        # Normalize to [-1.0, 1.0]
        max_val = np.max(np.abs(rotated))
        if max_val == 0.0:
            max_val = 1.0
        normalized = rotated / max_val
        
        # Find nearest codebook index
        indices = np.zeros_like(normalized, dtype=np.int8)
        for i in range(self.num_levels - 1):
            midpoint = (self.codebook[i] + self.codebook[i+1]) / 2.0
            indices[normalized > midpoint] = i + 1
            
        # Reconstruct (decompress)
        reconstructed_normed = self.codebook[indices]
        reconstructed = reconstructed_normed * max_val
        
        # 4. De-rotate back to original basis
        decompressed = np.dot(reconstructed, Q.T)
        
        # Memory calculation: 3 bits per element vs 64 bits (float64)
        compressed_memory = (matrix.size * self.target_bits) / 8.0 + 8.0 # bits to bytes + scaling factor
        memory_reduction = original_memory / compressed_memory
        
        # Reconstruction Error (MSE)
        mse = float(np.mean((matrix - decompressed) ** 2))
        
        return decompressed, memory_reduction, mse
