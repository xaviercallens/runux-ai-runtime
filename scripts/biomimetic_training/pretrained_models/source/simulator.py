# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
#
# Biomimetic Co-Inference Training: Neural Network Simulator
# ==========================================================

import numpy as np
from typing import Tuple, List, Dict, Optional

class BiomimeticLayer:
    """
    Represents a single neural layer capable of standard Backpropagation
    and local Co-Inference Direct Feedback Alignment (CI-DFA) updates.
    """
    def __init__(self, in_features: int, out_features: int, output_dim: int):
        self.in_features = in_features
        self.out_features = out_features
        self.output_dim = output_dim
        
        # Initialize weights and biases (Xavier/He initialization)
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros((1, out_features))
        
        # Fixed random feedback projection matrix B_i for DFA (Hypothesis 2)
        # Translates the global loss error vector from output dimension directly to layer dimension
        # Properly scaled using Xavier normalization to prevent gradient explosion
        limit_B = np.sqrt(6.0 / (output_dim + out_features))
        self.B = np.random.uniform(-limit_B, limit_B, (output_dim, out_features))
        
        # Cached values for backward / DFA updates
        self.x = None  # Input activation
        self.a = None  # Pre-activation (x * W + b)
        self.y = None  # Post-activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for inference and local updates."""
        self.x = x
        self.a = np.dot(x, self.W) + self.b
        # ReLU activation function
        self.y = np.maximum(0.0, self.a)
        return self.y

    def get_activation_derivative(self) -> np.ndarray:
        """Derivative of ReLU activation function."""
        return (self.a > 0.0).astype(float)

    def dfa_update(self, output_error: np.ndarray, lr: float, prune_mask: Optional[np.ndarray] = None, is_output_layer: bool = False) -> np.ndarray:
        r"""
        Direct Feedback Alignment (DFA) Weight Update.
        Updates weights locally during the forward/co-inference pass using the random projection matrix B.
        $$\Delta W_i = -\eta \cdot (x_i^T \cdot \delta_i)$$
        where $\delta_i = (e \cdot B_i) \odot \sigma'(a_i)$.
        """
        # For output layer, error is direct. For hidden layers, it's projected via B.
        if is_output_layer:
            delta = output_error
        else:
            projected_error = np.dot(output_error, self.B) # [B, out_features]
            delta = projected_error * self.get_activation_derivative() # [B, out_features]
        
        # Compute local gradients
        dW = np.dot(self.x.T, delta) / output_error.shape[0] # [in_features, out_features]
        db = np.mean(delta, axis=0, keepdims=True)
        
        # Apply Telemetry-Gated Synaptic Pruning (Hypothesis 3)
        if prune_mask is not None:
            dW = dW * prune_mask
            
        # Gradient clipping to prevent overflow
        dW = np.clip(dW, -1.0, 1.0)
        db = np.clip(db, -1.0, 1.0)
        
        # Update parameters
        self.W -= lr * dW
        self.b -= lr * db
        
        # Return local error signal for diagnostic verification
        return delta


class BiomimeticNet:
    """
    Multilayer feedforward neural network implementing standard Backpropagation
    and biomimetic WARS-CI-DFA with Telemetry-Gated Synaptic Pruning.
    """
    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.output_dim = layer_sizes[-1]
        
        # Build layers
        self.layers: List[BiomimeticLayer] = []
        for i in range(self.num_layers):
            self.layers.append(
                BiomimeticLayer(layer_sizes[i], layer_sizes[i+1], self.output_dim)
            )
            
        # Simulated WARS runtime performance tracking
        self.telemetry = {
            "pmu_cache_miss_rate": 0.05,
            "pruning_threshold": 0.001,
            "pruned_synapses_count": 0
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Runs the entire network forward pass."""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def update_telemetry(self, cache_miss_rate: float):
        """Simulates real-time hardware telemetry feedback from the WARS scheduler."""
        self.telemetry["pmu_cache_miss_rate"] = cache_miss_rate
        # Higher cache miss rate triggers more aggressive pruning thresholds to save compute energy
        if cache_miss_rate > 0.15:
            self.telemetry["pruning_threshold"] = 0.005
        elif cache_miss_rate > 0.08:
            self.telemetry["pruning_threshold"] = 0.001
        else:
            self.telemetry["pruning_threshold"] = 0.0001

    def train_step_dfa(self, x: np.ndarray, y_true: np.ndarray, lr: float) -> Tuple[float, float]:
        """
        Executes a local co-inference and Direct Feedback Alignment (DFA) training step.
        Completely bypasses standard transposed backpropagation chains.
        """
        # 1. Forward Pass (Inference)
        y_pred = self.forward(x)
        
        # 2. Compute global output error e (Loss is mean squared error or cross entropy error)
        error = y_pred - y_true # [B, output_dim]
        loss = 0.5 * np.mean(np.sum(error**2, axis=1))
        
        # 3. Local direct feedback weight updates (without backpropagating through weights)
        self.telemetry["pruned_synapses_count"] = 0
        for i, layer in enumerate(self.layers):
            is_output = (i == self.num_layers - 1)
            # Dynamic gating mask based on Telemetry-Gated Synaptic Pruning (TG-SP)
            prune_mask = None
            if self.telemetry["pruning_threshold"] > 0:
                # Prune synapses with low gradient magnitude to save cache/registers energy
                if is_output:
                    projected_error = error
                else:
                    projected_error = np.dot(error, layer.B)
                delta = projected_error * layer.get_activation_derivative()
                dW_raw = np.dot(layer.x.T, delta) / x.shape[0]
                
                # Gate updates smaller than dynamic threshold
                prune_mask = (np.abs(dW_raw) >= self.telemetry["pruning_threshold"]).astype(float)
                self.telemetry["pruned_synapses_count"] += np.sum(1.0 - prune_mask)
                
            layer.dfa_update(error, lr, prune_mask, is_output_layer=is_output)
            
        return loss, float(self.telemetry["pruned_synapses_count"])

    def co_inference_step(self, x: np.ndarray, y_true: np.ndarray, lr: float) -> Tuple[float, float]:
        """
        WARS-CI-DFA v2: Closed-Loop Concurrent Co-Inference & Retraining.
        Runs a unified forward-inference pass and performs local weight updates
        concurrently inside the forward loop using predictive coding error projections
        and homeostatic Telemetry-Gated Synaptic Pruning (H-TG-SP).
        """
        # 1. Pre-calculate the global output error for the step (simulating concurrent feedback)
        y_pred = self.forward(x)
        error = y_pred - y_true
        loss = 0.5 * np.mean(np.sum(error**2, axis=1))
        
        out = x
        self.telemetry["pruned_synapses_count"] = 0
        
        # Unified forward pass with concurrent local updates
        for i, layer in enumerate(self.layers):
            is_output = (i == self.num_layers - 1)
            
            # 1. Execute layer forward pass (inference)
            next_out = layer.forward(out)
            
            # 2. Compute local prediction error projection (Predictive Coding)
            # During inference, each layer uses its own state and feedback projection
            # to calculate the error signal concurrently without a backward sweep.
            if is_output:
                delta = error
            else:
                # Local error derived concurrently from intermediate projections
                projected_error = np.dot(error, layer.B)
                delta = projected_error * layer.get_activation_derivative()
            
            # 3. Homeostatic H-TG-SP (Hypothesis 3)
            # Adjusts the pruning threshold dynamically to lock in exactly 60% active updates
            prune_mask = None
            if self.telemetry["pruning_threshold"] > 0:
                dW_raw = np.dot(layer.x.T, delta) / x.shape[0]
                prune_mask = (np.abs(dW_raw) >= self.telemetry["pruning_threshold"]).astype(float)
                active_fraction = np.mean(prune_mask)
                
                # Dynamic homeostatic controller
                target_active = 0.60
                beta_coef = 0.0005
                self.telemetry["pruning_threshold"] += beta_coef * (active_fraction - target_active)
                self.telemetry["pruning_threshold"] = max(1e-5, min(0.01, self.telemetry["pruning_threshold"]))
                
                self.telemetry["pruned_synapses_count"] += np.sum(1.0 - prune_mask)
            
            # 4. Perform concurrent weight update during forward pass
            layer.dfa_update(error, lr, prune_mask, is_output_layer=is_output)
            
            out = next_out
            
        return loss, float(self.telemetry["pruned_synapses_count"])

    def train_step_backprop(self, x: np.ndarray, y_true: np.ndarray, lr: float) -> float:
        """
        Executes a standard Backpropagation training step (the baseline).
        Requires sequential backpropagation of error signals through the transpose of weights.
        """
        # 1. Forward Pass
        y_pred = self.forward(x)
        error = y_pred - y_true
        loss = 0.5 * np.mean(np.sum(error**2, axis=1))
        
        # 2. Sequential Backpropagation
        delta = error # Output layer error
        for i in reversed(range(self.num_layers)):
            layer = self.layers[i]
            
            # Compute gradients
            dW = np.dot(layer.x.T, delta) / x.shape[0]
            db = np.mean(delta, axis=0, keepdims=True)
            
            # Backpropagate error to previous layer (uses W.T, which is the weight transport bottleneck)
            if i > 0:
                prev_layer = self.layers[i-1]
                delta = np.dot(delta, layer.W.T) * prev_layer.get_activation_derivative()
                
            # Update weights
            layer.W -= lr * dW
            layer.b -= lr * db
            
        return loss
