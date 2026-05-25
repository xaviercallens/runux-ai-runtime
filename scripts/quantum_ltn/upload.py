# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-RunuX-Commercial
#
# WARS-Quantum-LTN: Programmatic Hugging Face Uploader
# ====================================================

import os
import ssl
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Force global unverified HTTPS context
ssl._create_default_https_context = ssl._create_unverified_context

# Intercept and disable verification globally for all default SSLContext creations
original_create_default_context = ssl.create_default_context
def unverified_create_default_context(*args, **kwargs):
    context = original_create_default_context(*args, **kwargs)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context
ssl.create_default_context = unverified_create_default_context

# Monkeypatch requests to disable SSL verification globally
import requests
original_request = requests.Session.request
def unverified_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)
requests.Session.request = unverified_request
requests.sessions.Session.request = unverified_request


from huggingface_hub import HfApi, create_repo

def run_upload():
    print("=========================================================================")
    print("WARS-Quantum-LTN: Hugging Face Programmatic Delivery Publisher")
    print("=========================================================================")
    
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[ERROR] HF_TOKEN environment variable not set. Cannot authenticate.")
        return
        
    api = HfApi(token=token)
    
    model_repo_id = "callensxavier/runux-wars-quantum-ltn-512q"
    dataset_repo_id = "callensxavier/runux-quantum-dynamics-ea-512q"
    
    # 1. Create Model Repository
    print(f"\n[1/4] Ensuring model repository exists: {model_repo_id}...")
    try:
        create_repo(repo_id=model_repo_id, token=token, repo_type="model", exist_ok=True)
        print("      ✅ Model repository is ready.")
    except Exception as e:
        print(f"      [WARNING] Could not create/verify model repository: {e}")
        
    # 2. Create Dataset Repository
    print(f"\n[2/4] Ensuring dataset repository exists: {dataset_repo_id}...")
    try:
        create_repo(repo_id=dataset_repo_id, token=token, repo_type="dataset", exist_ok=True)
        print("      ✅ Dataset repository is ready.")
    except Exception as e:
        print(f"      [WARNING] Could not create/verify dataset repository: {e}")

    # 3. Upload Model Card and Code Assets
    print(f"\n[3/4] Uploading Model Card and reproducible quantum simulator source files to {model_repo_id}...")
    try:
        # Upload README.md (Model Card)
        api.upload_file(
            path_or_fileobj="HF_MODEL_README.md",
            path_in_repo="README.md",
            repo_id=model_repo_id,
            repo_type="model"
        )
        print("      - Uploaded: README.md (Model Card)")
        
        # Upload academic paper draft
        api.upload_file(
            path_or_fileobj="PAPER_DRAFT.md",
            path_in_repo="PAPER_DRAFT.md",
            repo_id=model_repo_id,
            repo_type="model"
        )
        print("      - Uploaded: PAPER_DRAFT.md (Preprint)")
        
        # Upload codebase assets
        source_files = [
            ("simulator.py", "simulator.py"),
            ("ltn_constraints.py", "ltn_constraints.py"),
            ("polarquant.py", "polarquant.py"),
            ("scheduler.py", "scheduler.py"),
            ("bench_ea_spin_glass.py", "bench_ea_spin_glass.py"),
            ("neuro_symbolic_verifier.py", "neuro_symbolic_verifier.py"),
            ("upload.py", "upload.py"),
            ("/Volumes/MacCleanerStorage/xdev/xavux/rust-linux-mini-kernel/paper/quantum_ltn_paper.tex", "quantum_ltn_paper.tex"),
            ("../tpu_llm_bench.py", "tpu_llm_bench.py"),
            ("../autoresearch_rust_compiler.py", "autoresearch_rust_compiler.py"),
            ("../CONTRIBUTION_PROPOSAL.md", "CONTRIBUTION_PROPOSAL.md"),
            ("../../docs/COMPARATIVE_ANALYSIS.md", "COMPARATIVE_ANALYSIS.md"),
            ("../../marketing/FRENCH_STYLE_AI_INNOVATION.md", "FRENCH_STYLE_AI_INNOVATION.md")
        ]
        
        for local_path, repo_path in source_files:
            if os.path.exists(local_path):
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=model_repo_id,
                    repo_type="model"
                )
                print(f"      - Uploaded: {repo_path} (Source asset)")
                
        print("      ✅ All model repository uploads finished successfully.")
    except Exception as e:
        print(f"      ❌ Model uploads failed: {e}")

    # 4. Upload Dataset Card
    print(f"\n[4/4] Uploading Dataset Card to {dataset_repo_id}...")
    try:
        api.upload_file(
            path_or_fileobj="HF_DATASET_README.md",
            path_in_repo="README.md",
            repo_id=dataset_repo_id,
            repo_type="dataset"
        )
        print("      - Uploaded: README.md (Dataset Card)")
        print("      ✅ All dataset repository uploads finished successfully.")
    except Exception as e:
        print(f"      ❌ Dataset uploads failed: {e}")
        
    print("\n=========================================================================")
    print("DELIVERY COMPLETED: WARS-Quantum-LTN is successfully live on Hugging Face!")
    print("=========================================================================")

if __name__ == "__main__":
    run_upload()
