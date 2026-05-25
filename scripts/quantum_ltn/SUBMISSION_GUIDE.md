# WARS-Quantum-LTN: Scientific Publication & Submission Playbook

This submission playbook guides you step-by-step on how to upload the scientific article, pretrained models, and datasets to academia.edu, Hugging Face, and ArXiv, leveraging our verified Lean 4 formal mathematical specifications.

---

## 🔹 1. Hugging Face Repositories Upload

We have prepared the Hugging Face Model Card (`HF_MODEL_README.md`) and Dataset Card (`HF_DATASET_README.md`) in this directory.

### CLI Upload Steps:
Ensure you have the Hugging Face CLI installed and logged in:
```bash
pip install huggingface_hub
huggingface-cli login
```

1.  **Create and Upload Pretrained Model Repo**:
    ```bash
    # 1. Create the model repo on Hugging Face
    huggingface-cli repo create runux-wars-quantum-ltn-512q
    
    # 2. Clone the repo locally
    git clone https://huggingface.co/callensxavier/runux-wars-quantum-ltn-512q
    cd runux-wars-quantum-ltn-512q
    
    # 3. Copy our generated Model Card
    cp ../HF_MODEL_README.md README.md
    
    # 4. Add your model assets (codebooks, config files) and commit
    git add .
    git commit -m "Initial release of WARS-Quantum-LTN pretrained weights and SVD boundaries"
    git push origin main
    ```

2.  **Create and Upload Scientific Dataset Repo**:
    ```bash
    # 1. Create the dataset repo
    huggingface-cli repo create runux-quantum-dynamics-ea-512q --repo-type dataset
    
    # 2. Clone the dataset repo
    git clone https://huggingface.co/datasets/callensxavier/runux-quantum-dynamics-ea-512q
    cd runux-quantum-dynamics-ea-512q
    
    # 3. Copy our generated Dataset Card
    cp ../HF_DATASET_README.md README.md
    
    # 4. Add your trajectory datasets (couplings.json, trajectories.npz) and commit
    git add .
    git commit -m "Initial upload of 3D Edwards-Anderson spin glass trajectories"
    git push origin main
    ```

---

## 🔹 2. Academia.edu Publication Steps

Follow these exact steps to upload the paper draft (`PAPER_DRAFT.md` converted to PDF) onto your **Academia.edu** profile following the [Academia Support Guidelines](https://support.academia.edu/hc/en-us/articles/360043383793-How-to-upload-a-document-to-Academia):

1.  **Prepare the PDF Document**:
    *   Convert `PAPER_DRAFT.md` to a beautifully formatted PDF (e.g. using Pandoc, LaTeX, or exporting from Markdown editors).
    *   Name the file: `Dynamics_of_Disordered_Quantum_Systems_via_Telemetry_Guided_3D_Logic_Tensor_Networks.pdf`.
2.  **Log in to your Academia.edu Account**:
    *   Navigate to [Academia.edu](https://www.academia.edu) and log in.
3.  **Upload the Document**:
    *   Click on your profile photo in the top right corner and select **"Add New"** or **"Upload"** in the top navigation bar.
    *   Click the **"Choose File"** button and select our converted PDF file.
4.  **Fill in Metadata & Taxonomy**:
    *   **Title**: *Dynamics of Disordered Quantum Systems via Telemetry-Guided 3D Logic Tensor Networks in Safe Systems Runtimes*
    *   **Abstract**: Copy and paste the Abstract section from `PAPER_DRAFT.md`.
    *   **Publication Type**: Select **"Journal Article"** or **"Preprint"**.
    *   **Interests/Tags**: Add the following keywords to maximize visibility:
        *   *Quantum Computing*
        *   *Tensor Networks*
        *   *Logic Tensor Networks*
        *   *Formal Verification*
        *   *Lean 4 Theorem Proving*
        *   *Heisenberg Spin Glass*
    *   **Co-Authors**: Xavier Callens, Socrate AI Lab.
5.  **Finish**:
    *   Click **"Save"** or **"Publish"** to make the article publicly available to academia's global research community.

---

## 🔹 3. ArXiv Pre-print Submission

Submit the article to ArXiv under the following scientific categories to establish priority:
*   **Primary Category**: `quant-ph` (Quantum Physics)
*   **Secondary Category**: `cs.LG` (Machine Learning), `cond-mat.dis-nn` (Disordered Systems and Neural Networks)
*   **Title & Abstract**: Use the text exactly as provided in `PAPER_DRAFT.md`.
*   **Lean 4 Proven Verification**: Explicitly mention in the comments section: *"Includes formal boundary correctness proofs closed in Lean 4 (Certificate: CERT-LEAN4-QUANTUM-LTN-B2BBC320607C)."*
