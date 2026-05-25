#!/usr/bin/env python3
"""
Upload scientific article to HuggingFace dataset repo.
Token via env var only — never stored on disk.
"""
import os
import sys
import tempfile

def main():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        import httpx
        original_httpx_init = httpx.Client.__init__
        def patched_httpx_init(self, *args, **kwargs):
            kwargs['verify'] = False
            original_httpx_init(self, *args, **kwargs)
        httpx.Client.__init__ = patched_httpx_init
    except ImportError:
        pass
    try:
        import requests
        original_requests_init = requests.Session.__init__
        def patched_requests_init(self, *args, **kwargs):
            original_requests_init(self, *args, **kwargs)
            self.verify = False
        requests.Session.__init__ = patched_requests_init
    except ImportError:
        pass
    
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN environment variable")
        sys.exit(1)
    
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    username = api.whoami()["name"]
    repo_id = f"{username}/runux-tpu-v5e-benchmarks"
    
    # Upload scientific article
    article_path = os.path.join(os.path.dirname(__file__), "scientific_article.md")
    if os.path.exists(article_path):
        api.upload_file(
            path_or_fileobj=article_path,
            path_in_repo="scientific_article.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print(f"✓ Uploaded scientific_article.md to {repo_id}")
    
    # Upload real TPU results if available
    results_path = os.path.expanduser("~/tpu_benchmark_results.json")
    if os.path.exists(results_path):
        api.upload_file(
            path_or_fileobj=results_path,
            path_in_repo="tpu_v5e_real_results.json",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print(f"✓ Uploaded real TPU results to {repo_id}")
    
    print(f"\nDataset: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()
