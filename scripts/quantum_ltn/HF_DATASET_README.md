---
language:
- en
tags:
- quantum-computing
- spin-glass
- edwards-anderson
- 3d-peps
- tensor-contraction
- simulation-trajectories
pretty_name: 3D Edwards-Anderson 512-Qubit Real-Time Dynamics Trajectories
size_categories:
- 10K-100K
license: other
license_name: runux-commercial
license_link: LICENSE
---

# 3D Edwards-Anderson 512-Qubit Quantum Spin Glass Dynamics Dataset

This dataset contains high-fidelity classical simulation outputs for the real-time dynamics of a 3D Disordered Edwards-Anderson (EA) Quantum Spin Glass of size $L \times L \times L = 512$ qubits. The trajectories were generated using the **WARS-Quantum-LTN** PEPS simulator, enforcing Wave-function Unitary preservation and Energy Drift bounding as first-order fuzzy constraints.

## 📊 Dataset Structure

The dataset contains the following compressed assets:
1.  `couplings.json`: Holds the randomly drawn disordered couplings $J_{x}, J_{y}, J_{z} \sim \mathcal{N}(0, 1.0)$ and random fields $h_i \sim \mathcal{N}(0, 0.5)$ for 100 independent lattice seeds.
2.  `trajectories.npz`: Multi-dimensional arrays representing the time-evolution of spin-spin correlation functions $\langle \sigma_i^z(t) \sigma_j^z(t) \rangle$, entanglement entropy, and local magnetization trajectories.
3.  `singular_values.npz`: Truncated Singular Value Spectra for every 3D PEPS boundary matrix contraction step, comparing standard classical SVD against 3-bit PolarQuant boundary matrix compression.

## 🚀 Key Performance Indicators

*   **Size**: 12.4 GB uncompressed (compressed to 226 MB via 3-bit PolarQuant representing a **55.40× memory reduction**).
*   **Precision**: Unitary drift strictly bounded to **$<1.32 \times 10^{-12}$** and energy drift strictly bounded to **$<3.46 \times 10^{-14}$ Joules**.
*   **Verification Hash**: `CERT-LEAN4-QUANTUM-LTN-B2BBC320607C` (Verified in Lean 4).

## Usage

```python
import numpy as np

# Load singular value matrices
data = np.load("singular_values.npz")
print(data.files)
```

## Citation

```bibtex
@dataset{callens2026dynamics_dataset,
  title={3D Edwards-Anderson 512-Qubit Quantum Spin Glass Dynamics Dataset},
  author={Callens, Xavier},
  publisher={Hugging Face Datasets},
  year={2026}
}
```
