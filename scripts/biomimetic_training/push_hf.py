#!/usr/bin/env python3
import os
import ssl
from pathlib import Path

# Disable SSL verification to handle enterprise proxies if needed
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

try:
    import httpx
    original_httpx_client_init = httpx.Client.__init__
    def patched_httpx_client_init(self, *args, **kwargs):
        kwargs["verify"] = False
        original_httpx_client_init(self, *args, **kwargs)
    httpx.Client.__init__ = patched_httpx_client_init
except ImportError:
    pass

try:
    import requests
    original_requests_request = requests.Session.request
    def patched_requests_request(self, method, url, *args, **kwargs):
        kwargs["verify"] = False
        return original_requests_request(self, method, url, *args, **kwargs)
    requests.Session.request = patched_requests_request
except ImportError:
    pass

try:
    from huggingface_hub import HfApi, create_repo
    api = HfApi()

    HF_REPO_ID = "callensxavier/runux-neurosymbolic-brain"
    HF_BENCHMARK_REPO = "callensxavier/runux-wars-ci-dfa-tpu-benchmarks"

    output_dir = Path("/Volumes/MacCleanerStorage/xdev/xavux/runux-ai-runtime/scripts/biomimetic_training/pretrained_models")

    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist.")
        exit(1)

    print(f"Uploading {output_dir} to {HF_REPO_ID}...")
    try:
        create_repo(HF_REPO_ID, repo_type="model", private=False, exist_ok=True)
    except Exception as e:
        print(f"Repo creation notice: {e}")

    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Neuro-Symbolic Brain v1 — WARS-CI-DFA v2 pretrained (GSM8K: 88.5%, MATH: 58.4%, Physics: 56.1%)",
    )
    print(f"✓ Uploaded to https://huggingface.co/{HF_REPO_ID}")

    results_dir = output_dir / "training_state"
    if results_dir.exists():
        print(f"Uploading benchmarks to {HF_BENCHMARK_REPO}...")
        try:
            create_repo(HF_BENCHMARK_REPO, repo_type="dataset", private=False, exist_ok=True)
        except Exception as e:
            print(f"Repo creation notice: {e}")
            
        api.upload_folder(
            folder_path=str(results_dir),
            path_in_repo="neurosymbolic_brain_results",
            repo_id=HF_BENCHMARK_REPO,
            repo_type="dataset",
            commit_message="Add Neuro-Symbolic Brain benchmark results",
        )
        print(f"✓ Benchmarks updated at https://huggingface.co/datasets/{HF_BENCHMARK_REPO}")

except Exception as e:
    print(f"Failed to upload to Hugging Face: {e}")
