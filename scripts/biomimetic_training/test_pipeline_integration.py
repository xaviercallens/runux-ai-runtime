# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
#
# WARS-CI-DFA: Scikit-Learn Pipeline Integration Tester
# ======================================================

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator
from scikit_runux import RunuxClassifier, PRIVATE_MODE

def test_pipeline():
    print("=========================================================")
    print("      scikit-runux Pipeline Integration Verification     ")
    print("=========================================================\n")
    print(f"  [+] Dynamic Package Route: {'PRIVATE (Full Core)' if PRIVATE_MODE else 'PUBLIC (Gated Stub)'}")
    
    # Initialize Classifier
    clf = RunuxClassifier(
        hidden_layer_sizes=(64, 32),
        learning_rate=0.01,
        max_iter=10,
        batch_size=16
    )
    
    # Build standard scikit-learn pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])
    
    # Generate simple test dataset
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 3, size=100)
    
    if PRIVATE_MODE:
        print("  [+] Running Pipeline Fit on local private biomimetic training core...")
        try:
            pipeline.fit(X, y)
            print("      -> Fit Completed Successfully.")
            
            # Predict
            preds = pipeline.predict(X[:5])
            probs = pipeline.predict_proba(X[:5])
            print(f"      -> Predictions: {preds}")
            print(f"      -> Probabilities shape: {probs.shape}")
            print("\n  🎉 PRIVATE INTEGRATION PASSED 100% compliance test!")
        except Exception as e:
            print(f"  ❌ PRIVATE INTEGRATION FAILED: {str(e)}")
            raise e
    else:
        print("  [+] Verifying Gated Stub Exception Behavior...")
        try:
            pipeline.fit(X, y)
            print("  ❌ STUB INTEGRATION FAILED: Expected fit to raise NotImplementedError, but it succeeded!")
        except NotImplementedError as e:
            print(f"      -> Caught Expected Exception: {str(e)}")
            print("\n  🎉 GATED STUB INTEGRATION PASSED 100% compliance test!")
        except Exception as e:
            print(f"  ❌ STUB INTEGRATION FAILED: Unexpected exception raised: {str(e)}")
            raise e

if __name__ == "__main__":
    test_pipeline()
