# Hyperparameters Chosen During Hyperparameter Tuning with Wandb

| Hyper‑parameter | Typical Search Space (used in the notebooks) | Why It Matters on a T4 GPU |
|---|---|---|
| **Learning Rate** (`5e‑6 – 2e‑4`) | Small values avoid catastrophic forgetting; larger values speed up convergence but can cause instability. | The T4’s limited memory forces LoRA adapters, which are very sensitive to LR. Too high diverges, too low slows sweeps. This range lets the Bayesian optimizer find a sweet spot within the ~2‑3 hour per‑run budget. |
| **Gradient Accumulation Steps** (`4` or `8`) | Accumulates gradients over multiple forward passes before an update. | With batch size 2 (max for 15 GB with 4‑bit quantisation), effective batch sizes become 8 or 16, improving gradient stability without exceeding memory limits. |
| **LoRA Dropout** (`0.05` or `0.1`) | Randomly drops a fraction of LoRA adapter connections during training. | Dropout adds no memory overhead; on a T4, where we train on a subsampled dataset (~12 k samples), 5‑10 % dropout prevents over‑fitting without slowing training. |
| **Weight Decay** (`0.01` or `0.05`) | L2 regularisation on all trainable parameters. | Penalises large weights, encouraging simpler models that generalise better. Stronger decay (`0.05`) can achieve comparable performance in fewer steps, crucial given the T4’s limited compute. |
| **LoRA Rank (`r`)** | Typical values `8`, `16`, `32`. | Larger rank increases capacity but consumes more VRAM. On a T4 we stay at `r = 8` or `16` to keep the adapter footprint < 200 MB. |
| **LoRA Alpha** | Typical values `16`, `32`. | Controls scaling of LoRA updates; higher alpha can compensate for a smaller rank without extra memory cost. |

### How the Search Space Aligns with the T4 Constraints

*   **Memory‑first design** – All notebooks use 4‑bit quantisation (bnb‑4bit) and LoRA adapters. This reduces the model’s memory footprint to ~4 GB, leaving ~11 GB for activations, optimizer states, and the extra tensors required by WandB logging.
*   **Dataset subsampling** – The notebooks default to `USE_FULL_DATASET = False`, pulling only ~15 % of the original 100 k samples (≈12 k). This cuts the number of training steps per epoch dramatically, keeping each sweep run under the ~2‑3 hour wall‑time limit of a free‑tier T4.
*   **Early stopping & evaluation metric** – The sweep metric is `eval/loss` (instead of `train/loss`). Early stopping after ~1500 steps prevents wasteful epochs once the validation loss plateaus, which is crucial when GPU time is billed per hour.
*   **Bayesian optimisation** – WandB’s Bayesian sweeps focus on a small, well‑curated search space (the four hyper‑parameters above). This yields high‑quality suggestions after only a handful of trials, ideal for the limited compute budget of a T4.

### Practical Tips When Running on a T4

| Tip                                                                                             | Reason                                                                                                                                |
| :---------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| Set `batch_size=2` and use gradient accumulation                                                | Guarantees the model fits in VRAM while still achieving an effective batch size of 8‑16.                                              |
| Enable `torch.cuda.empty_cache()` before each sweep run                                         | Frees any fragmented memory left by previous runs, avoiding OOM errors.                                                               |
| Pin the GPU (`CUDA_VISIBLE_DEVICES=0`) and limit the number of parallel workers in WandB (`--max-concurrency 1`) | Prevents the T4 from being overloaded by multiple simultaneous processes.                                                             |
| Log only essential metrics (e.g., `eval/loss`, `learning_rate`)                                 | Reduces WandB overhead and keeps the run lightweight.                                                                                 |
| Use mixed‑precision (`fp16`) together with 4‑bit quantisation                                   | Further cuts memory usage and speeds up each step.                                                                                    |

Summary
Learning rate, gradient accumulation, LoRA dropout, and weight decay form a compact, GPU‑friendly search space that respects the T4’s 15 GB VRAM limit.
The chosen ranges let the Bayesian optimiser explore enough diversity to find a performant configuration without exhausting the GPU or exceeding the typical free‑tier runtime budget.
Additional knobs like LoRA rank/alpha can be tuned if you have extra headroom, but the defaults (r = 8, alpha = 16) already provide a good trade‑off between capacity and memory.