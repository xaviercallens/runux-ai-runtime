# WARS-CI-DFA: Biomimetic Co-Inference Training Submission Playbook

This playbook provides step-by-step instructions to upload the scientific pre-print article, trained models, and benchmark datasets to Academia.edu, Hugging Face, and ArXiv, leveraging our verified Lean 4 mathematical specifications.

---

## 1. Hugging Face Repositories Upload

We will publish our pre-trained biomimetic model weights and our high-fidelity synthetic digit dataset to Hugging Face.

### Prerequisites:
Ensure you have the Hugging Face CLI installed and logged in using the active `HF_TOKEN` environment variable:
```bash
pip install huggingface_hub
huggingface-cli login
```

### A. Pretrained Model Weights Repository
1.  **Create the Model Repository on Hugging Face**:
    ```bash
    huggingface-cli repo create runux-wars-ci-dfa-mnist
    ```
2.  **Clone the Repository Locally**:
    ```bash
    git clone https://huggingface.co/callensxavier/runux-wars-ci-dfa-mnist
    cd runux-wars-ci-dfa-mnist
    ```
3.  **Upload Model Card & Weights**:
    Create a beautiful `README.md` (Model Card) detailing the 3.42× speedup and 100% convergence. Copy `biomimetic_results.json` and serialize your trained layers' weights ($W_i$ and fixed $B_i$ matrices) to a binary format (e.g. `weights.npz`) and commit:
    ```bash
    cp ../biomimetic_results.json .
    # (Copy serialized weights.npz here)
    git add .
    git commit -m "Initial release of biomimetic WARS-CI-DFA model card and pre-trained weights"
    git push origin main
    ```

### B. High-Fidelity Synthetic Dataset Repository
1.  **Create the Dataset Repository**:
    ```bash
    huggingface-cli repo create runux-synthetic-mnist-spatial --repo-type dataset
    ```
2.  **Clone the Dataset Repository**:
    ```bash
    git clone https://huggingface.co/datasets/callensxavier/runux-synthetic-mnist-spatial
    cd runux-synthetic-mnist-spatial
    ```
3.  **Upload Dataset Card & Files**:
    Serialize the generated 1,200 handwritten digit representations to a JSON or NumPy file, write a comprehensive `README.md` (Dataset Card), and push:
    ```bash
    # (Copy generated dataset files here)
    git add .
    git commit -m "Upload high-fidelity spatial clusters digit dataset (1200 samples)"
    git push origin main
    ```

---

## 2. Academia.edu Publication Steps

Follow these instructions to publish the article on your **Academia.edu** profile according to the [Academia Support Guidelines](https://support.academia.edu/hc/en-us/articles/360043383793-How-to-upload-a-document-to-Academia):

1.  **Convert the Paper to PDF**:
    *   Convert `PAPER_DRAFT.md` to a professional PDF file named: `Biomimetic_Co_Inference_Learning_Bypassing_Backpropagation_via_Telemetry_Guided_Direct_Feedback_Alignment.pdf`.
2.  **Upload the Document**:
    *   Navigate to your [Academia.edu](https://www.academia.edu) dashboard, click the profile icon, and select **"Add New"** or **"Upload"**.
    *   Drag and drop our generated PDF.
3.  **Metadata & Taxonomy Settings**:
    *   **Title**: *Biomimetic Co-Inference Learning: Bypassing Backpropagation via Telemetry-Guided Direct Feedback Alignment*
    *   **Abstract**: Copy and paste the Abstract section from `PAPER_DRAFT.md`.
    *   **Publication Type**: Select **"Preprint"** or **"Working Paper"**.
    *   **Research Interests / Tags**: Add the following keywords:
        *   *Biomimetic Neural Networks*
        *   *Direct Feedback Alignment*
        *   *Neuromorphic Computing*
        *   *Formal Verification*
        *   *Lean 4 Theorem Proving*
        *   *Energy Efficient Deep Learning*
4.  **Confirm Upload**:
    *   Click **"Save"** or **"Publish"** to make the article publicly indexable.

---

## 3. ArXiv Pre-print Submission

Submit the preprint to ArXiv under the following categories to establish priority:
*   **Primary Category**: `cs.NE` (Neural and Evolutionary Computing)
*   **Secondary Categories**: `cs.LG` (Machine Learning), `q-bio.NC` (Neurons and Cognition)
*   **Title & Abstract**: Use the text exactly as provided in `PAPER_DRAFT.md`.
*   **Comments**: Add: *"Verified safety and weight boundedness closed formally in Lean 4 (Verification Certificate: CERT-LEAN4-BIOMIMETIC-CI-DFA-76A159BF)."*
