# Latent Shaper

**Advanced toolkit for analyzing, processing, and merging LoRAs for Z-Image Turbo (S3-DiT) and SDXL architectures.**

This project provides both a standalone **Gradio GUI** for deep model inspection/editing and a suite of **ComfyUI Custom Nodes** for complex pipeline workflows.

---

## Standalone GUI Studio

A powerful interface for surgical LoRA manipulation without loading ComfyUI. Features **Zero-Copy** workflow (direct disk access).

| Tab | Functionality |
| :--- | :--- |
| **Analyze** | **Deep Spectral Analysis**. View singular value spectrum, layer heatmaps, and get automated fixes via **AI Advisor**. |
| **Extract** | Extract LoRAs from checkpoints (`Tuned - Base`) with **Noise Gating** and BF16 precision support. |
| **Morph** | **Weight Equalizer**. Apply region-based EQ (IN/MID/OUT), Spectral Filters (SVD), DARE (Random Drop), and Block Erasers. |
| **Resize** | Smart Rank reduction using SVD to compress file size while retaining information. |
| **Merge** | Multi-model merging with advanced algorithms: **SVD (Smart Blend)**, **TIES**, **Geometric Median**, and **Orthogonal**. |
| **Utils** | **Maintenance Tools**. Convert precision (FP32/BF16), Normalize Keys (Kohya-ss format), and Rescale Alpha/Rank. |

### How to Launch GUI
```bash
# Inside the app folder
python gui_launcher.py
```

---

## ComfyUI Custom Nodes

Build complex processing pipelines directly in your graph.

### Pipeline Nodes
*   **LS Loader (Raw):** Loads LoRA directly into RAM for processing (bypassing standard Comfy loading).
*   **LS EQ (Structure):** Apply gain/attenuation to Input, Middle, or Output blocks.
*   **LS Filters (Signal):** Apply FFT (Frequency) or Band-Stop filters to remove noise.
*   **LS Dynamics (Rank):** SVD Spectral Gating and DARE (Drop And REscale) operations.
*   **LS Eraser (Block):** Surgically remove specific blocks or semantic concepts (via CLIP).
*   **LS Metadata:** Edit internal metadata, trigger words, and model names.
*   **LS Analyzer:** Output visual heatmaps of layer energy to the image window.
*   **LS Save:** Save the processed `Z_LORA` to disk (`.safetensors`).
*   **LS Apply:** Apply the processed `Z_LORA` to a Model/CLIP stream.

### Merging Nodes
*   **LS Merger (6-Input):** Merge up to 6 LoRAs simultaneously with weighted ratios and selectable algorithms (SVD, TIES, SLERP, etc.).

---

## ComfyUI Installation

1.  Navigate to your ComfyUI custom nodes directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone the repository:
    ```bash
    git clone https://github.com/Einhorn13/LatentShaper.git
    ```
3.  Install requirements:
    ```bash
    cd LatentShaper
    pip install -r requirements.txt
    ```
4.  **Restart ComfyUI**.

---

## Configuration
On the first launch of the GUI, go to the **Settings** tab to configure your `checkpoints` and `loras` directories for direct disk access.

---

## Standalone GUI Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Einhorn13/LatentShaper.git
    ```
2.  Install requirements:
    ```bash
    cd LatentShaper
    pip install -r requirements.txt
    ```
3.  ```bash
# Start GUI
python gui_launcher.py
```

## ComfyUI screenshot
<img width="1685" height="1055" alt="image" src="https://github.com/user-attachments/assets/87294777-85a9-470b-b3fc-3ff27fbd3bea" />

## GUI screenshot
<img width="1892" height="837" alt="image" src="https://github.com/user-attachments/assets/69ec4734-0d14-4ce5-961e-09973e872138" />




