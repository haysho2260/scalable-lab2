---
title: Scalable Lab2
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
---


# Scalable Lab 2 Chatbot

An example chatbot built with [Gradio](https://gradio.app), utilizing [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index) and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).

## 🔗 Project Resources
- **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/hayitsmaddy/scalable_lab2)
- **Checkpoints:** [Google Drive Folder](https://drive.google.com/drive/folders/16vEyHThVWbIrFUZtNZr2T7uEAb4Akp59?usp=sharing)

## 🧠 Model Training & Hyperparameters
Hyperparameter tuning was conducted using **Weights & Biases (WandB)**. A parameter sweep was run on the baseline model to identify the optimal configuration, which was then used to train the `model_centric` model.

**Training Configuration:**
- **Batch Size:** Per-device train batch size of 2 with 8 gradient accumulation steps.
- **Learning Rate:** 5e-5 with a cosine scheduler and a warmup ratio of 0.03.
- **Optimizer:** 8-bit AdamW (`adamw_8bit`) with weight decay of 0.01 and max gradient norm of 1.0.
- **Precision:** Mixed precision training (BF16 if supported, otherwise FP16).
- **Duration:** 1 epoch.
- **Logging:** Metrics reported to WandB.

## 💾 Checkpointing
Checkpointing was managed via `TrainingArguments` in the `SFTTrainer` (see `models/baseline.ipynb`):
- **Strategy:** `steps` (save based on step count, not epochs).
- **Frequency:** Save every 100 steps.
- **Retention:** Keep only the 3 most recent checkpoints to conserve storage (`save_total_limit=3`).

## ⚡ Performance & Latency
You may observe that the model running on this Gradio interface is slower than on a machine with a dedicated GPU.

- **CPU vs GPU:** This deployment currently runs on **CPU** (or uses `n_gpu_layers=0`), processing complex matrix calculations sequentially.
- **GPU Speedup:** Running on a GPU (NVIDIA T4, A100, or Mac Metal) enables massive parallelization, resulting in near-instant token generation.
- **Local Optimization:** To improve local performance, install `llama-cpp-python` with hardware acceleration (e.g., Metal for macOS, CUDA for Linux) and set `n_gpu_layers=-1` in `app.py`.