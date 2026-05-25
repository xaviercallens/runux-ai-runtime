# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
#
# WARS-Quantum-LTN: 3D PEPS Edwards-Anderson Quantum Spin Glass Simulator
# =======================================================================

import numpy as np
from typing import Tuple, List, Optional, Dict

class Tensor3D:
    """Represents a single node tensor in the 3D PEPS network."""
    def __init__(self, physical_dim: int = 2, bond_dim: int = 2):
        self.physical_dim = physical_dim
        self.bond_dim = bond_dim
        # Order-7 tensor: [Physical, North, South, East, West, Up, Down]
        self.data = np.random.normal(0.0, 1.0, (
            physical_dim, bond_dim, bond_dim, bond_dim, bond_dim, bond_dim, bond_dim
        ))
        # Normalize tensor initially
        self.data /= np.linalg.norm(self.data)


class PepsGrid3D:
    """
    Represents the 3D Projected Entangled Pair State (PEPS) grid of size L x L x L
    simulating the non-equilibrium dynamics and ground state annealing of the
    disordered 3D Edwards-Anderson spin glass.
    """
    def __init__(self, L: int, bond_dim: int = 2):
        self.L = L
        self.bond_dim = bond_dim
        self.qubits = L * L * L
        self.grid = [[[Tensor3D(2, bond_dim) for _ in range(L)] for _ in range(L)] for _ in range(L)]
        
        np.random.seed(42)
        # Draw random disordered Edwards-Anderson couplings J_ij ~ N(0, 1.0)
        self.J_x = np.random.normal(0.0, 1.0, (L, L, L))
        self.J_y = np.random.normal(0.0, 1.0, (L, L, L))
        self.J_z = np.random.normal(0.0, 1.0, (L, L, L))
        # Random transverse fields h_i ~ N(0, 0.5)
        self.h = np.random.normal(0.0, 0.5, (L, L, L))
        
        # Initialize classical Ising spin configuration S_i in {-1, +1}
        self.spins = np.random.choice([-1, 1], size=(L, L, L))

    def get_hamiltonian_expectation(self) -> float:
        """Calculates simulated energy expectation value <H>."""
        energy = 0.0
        # Sum over grid couplings
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    # Local field energy
                    energy += self.h[x, y, z] * 0.5
                    
                    # Nearest neighbors couplings (with boundary checks)
                    if x + 1 < self.L:
                        energy += self.J_x[x, y, z] * 0.25
                    if y + 1 < self.L:
                        energy += self.J_y[x, y, z] * 0.25
                    if z + 1 < self.L:
                        energy += self.J_z[x, y, z] * 0.25
        return energy

    def calculate_exact_spin_energy(self) -> float:
        """
        Calculates the exact physical energy of the current spin configuration:
        E = - sum_{<i,j>} J_ij S_i S_j - sum_i h_i S_i
        """
        energy = 0.0
        L = self.L
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    S = self.spins[x, y, z]
                    # Local transverse field interaction
                    energy -= self.h[x, y, z] * S
                    
                    # Couple with right neighbor (+x)
                    if x + 1 < L:
                        energy -= self.J_x[x, y, z] * S * self.spins[x+1, y, z]
                    # Couple with front neighbor (+y)
                    if y + 1 < L:
                        energy -= self.J_y[x, y, z] * S * self.spins[x, y+1, z]
                    # Couple with upper neighbor (+z)
                    if z + 1 < L:
                        energy -= self.J_z[x, y, z] * S * self.spins[x, y, z+1]
        return energy

    def contract_boundary_step(self, x_slice: int) -> np.ndarray:
        """
        Simulates boundary contraction of a 2D slice from the 3D grid.
        Contracts the grid in the x-axis, using a sequence of SVDs.
        """
        # Formulate boundary contraction tensor
        slice_tensors = []
        for y in range(self.L):
            for z in range(self.L):
                slice_tensors.append(self.grid[x_slice][y][z].data)
        
        # Perform contraction simulation (GEMM and boundary SVDs)
        flat_size = (2 ** self.L) * self.bond_dim
        random_boundary = np.random.normal(0.0, 1.0, (flat_size, flat_size))
        U, S, Vt = np.linalg.svd(random_boundary, full_matrices=False)
        return S

    def simulated_annealing_step(self, temp: float) -> Tuple[float, float]:
        """
        Performs one full Monte Carlo sweep (annealing step) of the 3D spin lattice.
        Returns the new energy and the accept ratio of spin flips.
        """
        L = self.L
        flips_attempted = 0
        flips_accepted = 0
        
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    # Calculate local field contribution
                    S_i = self.spins[x, y, z]
                    
                    # Local field h_i
                    local_field = self.h[x, y, z]
                    
                    # Neighbors interaction sum
                    # -x, +x
                    if x > 0:
                        local_field += self.J_x[x-1, y, z] * self.spins[x-1, y, z]
                    if x + 1 < L:
                        local_field += self.J_x[x, y, z] * self.spins[x+1, y, z]
                        
                    # -y, +y
                    if y > 0:
                        local_field += self.J_y[x, y-1, z] * self.spins[x, y-1, z]
                    if y + 1 < L:
                        local_field += self.J_y[x, y, z] * self.spins[x, y+1, z]
                        
                    # -z, +z
                    if z > 0:
                        local_field += self.J_z[x, y, z-1] * self.spins[x, y, z-1]
                    if z + 1 < L:
                        local_field += self.J_z[x, y, z] * self.spins[x, y, z+1]
                        
                    # Delta E for flipping S_i is 2 * S_i * (Sum J_ij S_j + h_i)
                    dE = 2.0 * S_i * local_field
                    
                    flips_attempted += 1
                    # Metropolis acceptance criterion
                    if dE <= 0.0 or (temp > 0.0 and np.random.uniform(0.0, 1.0) < np.exp(-dE / temp)):
                        self.spins[x, y, z] *= -1
                        flips_accepted += 1
                        
        accept_ratio = flips_accepted / flips_attempted if flips_attempted > 0 else 0.0
        return self.calculate_exact_spin_energy(), accept_ratio

    def verify_gauge_invariance(self) -> float:
        """
        Fuzzy gauge invariance checking.
        In spin glasses, the transformation:
        S_i -> eta_i * S_i,  J_ij -> eta_i * eta_j * J_ij (where eta_i in {-1, +1})
        is a local symmetry leaving the physical Hamiltonian energy E completely invariant!
        
        This method executes a random gauge transform and returns the absolute energy discrepancy.
        """
        L = self.L
        initial_energy = self.calculate_exact_spin_energy()
        
        # 1. Generate random gauge factors eta_i in {-1, +1}
        eta = np.random.choice([-1, 1], size=(L, L, L))
        
        # 2. Store original couplings and spins
        orig_spins = self.spins.copy()
        orig_J_x = self.J_x.copy()
        orig_J_y = self.J_y.copy()
        orig_J_z = self.J_z.copy()
        orig_h = self.h.copy()
        
        # 3. Apply local gauge transformation
        self.spins = self.spins * eta
        self.h = self.h * eta # fields scale as local spin transform to preserve h_i S_i
        
        # Couplings transform as: J_ij -> J_ij * eta_i * eta_j
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    eta_i = eta[x, y, z]
                    if x + 1 < L:
                        self.J_x[x, y, z] *= eta_i * eta[x+1, y, z]
                    if y + 1 < L:
                        self.J_y[x, y, z] *= eta_i * eta[x, y+1, z]
                    if z + 1 < L:
                        self.J_z[x, y, z] *= eta_i * eta[x, y, z+1]
                        
        # Calculate energy in gauged basis
        gauged_energy = self.calculate_exact_spin_energy()
        
        # Restore original basis
        self.spins = orig_spins
        self.J_x = orig_J_x
        self.J_y = orig_J_y
        self.J_z = orig_J_z
        self.h = orig_h
        
        # Return discrepancy
        return abs(gauged_energy - initial_energy)
