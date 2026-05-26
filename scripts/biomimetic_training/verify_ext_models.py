# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
#
# WARS-CI-DFA v2: Comprehensive Extension Validator
# ===================================================

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scikit_runux.scikit_runux_ext import RunuxClassifier, RunuxRegressor, RunuxAutoEncoder

def verify_all_models():
    print("========================================================================")
    print("       WARS-CI-DFA v2: Scikit-Learn Extension Multi-Model Validator      ")
    print("========================================================================\n")

    # 1. Validate RunuxClassifier (Classification & Concurrent Co-Inference & Retraining)
    print("  [+] Step 1: Validating RunuxClassifier...")
    np.random.seed(42)
    X_cls = np.random.randn(150, 8)
    y_cls = np.random.randint(0, 3, size=150)
    
    clf = RunuxClassifier(hidden_layer_sizes=(32, 16), learning_rate=0.01, max_iter=5, batch_size=16)
    clf.fit(X_cls, y_cls)
    print(f"      -> Standard fit succeeded. Classes learned: {clf.classes_}")
    
    # Online streaming test with partial_fit
    clf.partial_fit(X_cls[:10], y_cls[:10])
    print("      -> partial_fit (Online Retraining) succeeded.")
    
    # Concurrent Co-Inference & Retraining test
    preds = clf.predict_and_partial_fit(X_cls[:5], y_cls[:5])
    print(f"      -> predict_and_partial_fit (Concurrent Co-Inference & Retraining) succeeded.")
    print(f"         Conformed Predictions: {preds}")

    # 2. Validate RunuxRegressor (Regression & Concurrent Co-Inference & Retraining)
    print("\n  [+] Step 2: Validating RunuxRegressor...")
    X_reg = np.random.randn(120, 6)
    y_reg = np.dot(X_reg, np.array([1.5, -2.0, 0.5, 0.0, 1.0, -0.5])) + np.random.randn(120) * 0.1
    
    reg = RunuxRegressor(hidden_layer_sizes=(32, 16), learning_rate=0.01, max_iter=5, batch_size=16)
    reg.fit(X_reg, y_reg)
    print("      -> Standard fit succeeded.")
    
    # Regression prediction
    reg_preds = reg.predict(X_reg[:5])
    print(f"      -> Predictions sample: {reg_preds}")
    
    # Concurrent Co-Inference & Retraining
    reg_co_preds = reg.predict_and_partial_fit(X_reg[:5], y_reg[:5])
    print(f"      -> Concurrent Co-Inference predictions sample: {reg_co_preds}")

    # 3. Validate RunuxAutoEncoder (Unsupervised compression & reconstruction)
    print("\n  [+] Step 3: Validating RunuxAutoEncoder...")
    X_ae = np.random.randn(200, 12)
    
    ae = RunuxAutoEncoder(hidden_layer_sizes=(8, 4), learning_rate=0.01, max_iter=10, batch_size=32)
    ae.fit(X_ae)
    print("      -> Unsupervised representation fit succeeded.")
    
    # Compression (encoder)
    Xt = ae.transform(X_ae[:5])
    print(f"      -> Latent compressed representation shape: {Xt.shape} (Input: {X_ae[:5].shape})")
    
    # Reconstruction (decoder)
    X_recon = ae.inverse_transform(Xt)
    print(f"      -> Reconstructed dimension shape: {X_recon.shape}")
    
    # Online representation update via partial_fit
    ae.partial_fit(X_ae[:10])
    print("      -> partial_fit (Online Unsupervised Learning) succeeded.")

    # 4. Pipeline Integration check
    print("\n  [+] Step 4: Verifying Scikit-Learn Pipeline integration...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])
    pipeline.fit(X_cls, y_cls)
    pipe_preds = pipeline.predict(X_cls[:5])
    print(f"      -> Pipeline execution completed successfully. Predictions: {pipe_preds}")
    
    print("\n  🎉 All WARS-CI-DFA v2 machine learning models passed validation successfully!")
    print("  ========================================================================")

if __name__ == "__main__":
    verify_all_models()
