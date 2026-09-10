#!/usr/bin/env python3
# ==============================================================================
# RunuX AI Runtime — Zenodo Open Science Deposition Script
#
# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# Licensed under CC-BY-4.0 (for public scientific data and paper).
# ==============================================================================

import os
import sys
import json
import argparse
from pathlib import Path
import urllib.request
import urllib.parse
import urllib.error


def publish_to_zenodo(
    token: str,
    bundle_path: Path,
    paper_path: Path,
    metadata_path: Path,
    sandbox: bool = False,
) -> dict:
    base_url = "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    print(f"Connecting to Zenodo ({'Sandbox' if sandbox else 'Production'})...")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    payload = json.dumps({"metadata": metadata}).encode("utf-8")

    # 1. Create deposition
    req = urllib.request.Request(
        f"{base_url}/deposit/depositions",
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            deposition = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        print(f"Error creating Zenodo deposition: {e.code} {err_msg}")
        return {"status": "ERROR", "error": err_msg}

    deposition_id = deposition["id"]
    bucket_url = deposition.get("links", {}).get("bucket")
    print(f"Created deposition ID: {deposition_id}")

    # 2. Upload files if bucket URL is present
    files_to_upload = [bundle_path, paper_path]
    for file_path in files_to_upload:
        if not file_path.exists():
            continue
        print(f"Uploading {file_path.name} to Zenodo...")
        file_size = file_path.stat().st_size
        with open(file_path, "rb") as f_data:
            if bucket_url:
                upload_url = f"{bucket_url}/{urllib.parse.quote(file_path.name)}"
                up_req = urllib.request.Request(
                    upload_url,
                    data=f_data,
                    headers={"Authorization": f"Bearer {token}"},
                    method="PUT",
                )
            else:
                upload_url = f"{base_url}/deposit/depositions/{deposition_id}/files"
                boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
                # Simple fallback
                up_req = urllib.request.Request(
                    upload_url,
                    data=f_data.read(),
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/octet-stream",
                    },
                    method="POST",
                )
            with urllib.request.urlopen(up_req) as up_resp:
                print(f"Uploaded {file_path.name} successfully.")

    record_url = deposition.get("links", {}).get("html", f"https://zenodo.org/deposit/{deposition_id}")
    doi = deposition.get("metadata", {}).get("prereserve_doi", {}).get("doi", f"10.5281/zenodo.{deposition_id}")

    print(f"Zenodo Deposition Ready!")
    print(f"  • DOI: {doi}")
    print(f"  • URL: {record_url}")

    return {
        "status": "SUCCESS",
        "deposition_id": deposition_id,
        "doi": doi,
        "url": record_url,
    }


def main():
    parser = argparse.ArgumentParser(description="Upload RunuX Open Science bundle to Zenodo")
    parser.add_argument("--token", type=str, default=os.environ.get("ZENODO_TOKEN"), help="Zenodo API access token")
    parser.add_argument("--bundle", type=str, default="zenodo_research_bundle_runux.tar.gz", help="Path to archive")
    parser.add_argument("--paper", type=str, default="papers/runux_scientific_proof_paper.pdf", help="Path to paper PDF")
    parser.add_argument("--metadata", type=str, default="zenodo.json", help="Path to zenodo.json")
    parser.add_argument("--sandbox", action="store_true", help="Use Zenodo Sandbox")
    args = parser.parse_args()

    if not args.token:
        print("ZENODO_TOKEN environment variable or --token argument required.")
        print("To publish, export ZENODO_TOKEN=<your_zenodo_token> and run: python3 scripts/publish_zenodo.py")
        sys.exit(0)

    publish_to_zenodo(
        token=args.token,
        bundle_path=Path(args.bundle),
        paper_path=Path(args.paper),
        metadata_path=Path(args.metadata),
        sandbox=args.sandbox,
    )


if __name__ == "__main__":
    main()
