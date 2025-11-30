# Scalable Lab Notebook - Modification History

This document tracks all modifications made to the original Unsloth fine-tuning notebook to add WandB hyperparameter sweeps, multi-environment support, and overfitting prevention.

---

## Table of Contents
1. [Original Notebook](#original-notebook)
2. [Project Goals](#project-goals)
3. [Modifications Made](#modifications-made)
4. [Final Configuration](#final-configuration)
5. [How to Use](#how-to-use)

---

## Original Notebook

**Source**: Unsloth DeepSeek-R1 fine-tuning template  
**Model**: `unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit`  
**Dataset**: `mlabonne/FineTome-100k` (100k instruction-following examples)  
**Original Training**: Single run with fixed hyperparameters

---

## Project Goals

1. ✅ Add **WandB integration** for experiment tracking
2. ✅ Implement **Bayesian hyperparameter optimization** using WandB Sweeps
3. ✅ Support **multiple environments** (Google Colab, Kaggle, Local)
4. ✅ Optimize for **Colab/Kaggle T4 GPU** free tier constraints
5. ✅ Prevent **overfitting** through regularization techniques
6. ✅ Enable **model upload** to Hugging Face Hub

---

## Modifications Made

### 1. Environment Detection & Authentication

**Changes**:
- Added automatic detection for Colab, Kaggle, and Local environments
- Implemented conditional token loading
- Added automatic login to Hugging Face and WandB

### 2. Dataset Subsampling for Free GPU Tiers

**Changes**:
- Added `USE_FULL_DATASET` toggle (default: `False`)
- Subsamples to 15% of data (~12k samples) for hyperparameter sweeps
- Reduces training time from ~30 hours to ~3-4 hours per epoch

### 3. WandB Bayesian Hyperparameter Sweep

**Hyperparameters Being Optimized**:
1. **Learning Rate**: 5e-6 to 2e-4
2. **Gradient Accumulation**: 4 or 8 steps
3. **LoRA Dropout**: 0.05 or 0.1
4. **Weight Decay**: 0.01 or 0.05

#### Why These Hyperparameters?

**Learning Rate (5e-6 to 2e-4)**
- **Lower bound (5e-6)**: Conservative, prevents catastrophic forgetting of base model knowledge
- **Upper bound (2e-4)**: Standard for LoRA fine-tuning, balances speed vs stability
- **Reduced from original**: Lowered from `1e-5 to 5e-4` because higher LRs caused overfitting
- **Reasoning**: LoRA adapters need smaller learning rates than full fine-tuning since they modify a pre-trained model

**Gradient Accumulation (4 or 8 steps)**
- **Effective batch size**: With `batch_size=2`, gives effective batch sizes of 8 or 16
- **Memory constraint**: T4 GPU (15GB) can't handle larger batch sizes
- **Sweet spot**: 8-16 is optimal for stable training without excessive memory
- **Reasoning**: Larger effective batch sizes → more stable gradients → better generalization

**LoRA Dropout (0.05 or 0.1)**
- **0.05 (5%)**: Light regularization, preserves most learning capacity
- **0.1 (10%)**: Moderate regularization, stronger overfitting prevention
- **Not higher**: >0.1 can hurt learning too much
- **Reasoning**: Dropout randomly disables connections during training, forcing the model to learn robust features instead of memorizing

**Weight Decay (0.01 or 0.05)**
- **0.01**: Standard L2 regularization strength
- **0.05**: Stronger regularization for aggressive overfitting prevention
- **Not higher**: >0.1 can prevent the model from learning effectively
- **Reasoning**: Weight decay penalizes large weights, encouraging simpler models that generalize better

#### Alternative Configurations

**If Still Overfitting** (More aggressive anti-overfitting):
```python
'parameters': {
    'learning_rate': {'min': 1e-6, 'max': 1e-4},
    'gradient_accumulation_steps': {'values': [8, 16]},
    'lora_dropout': {'values': [0.1, 0.15, 0.2]},
    'weight_decay': {'values': [0.05, 0.1]},
    'lora_r': {'values': [8, 16]},
}
```

**If Training Too Slow** (Faster convergence):
```python
'parameters': {
    'learning_rate': {'min': 1e-5, 'max': 3e-4},
    'gradient_accumulation_steps': {'values': [4]},
    'lora_dropout': {'values': [0.05]},
    'weight_decay': {'values': [0.01]},
    'warmup_ratio': {'values': [0.03, 0.06]},
}
```

**Architecture Tuning** (Advanced):
```python
'parameters': {
    'learning_rate': {'min': 5e-6, 'max': 2e-4},
    'lora_r': {'values': [8, 16, 32]},
    'lora_alpha': {'values': [16, 32]},
    'lora_dropout': {'values': [0.05, 0.1]},
}
```

### 4. Overfitting Prevention

**Changes**:
- Changed optimization metric from `train/loss` to `eval/loss`
- Added validation dataset evaluation
- Implemented early stopping at 1500 steps
- Added LoRA dropout and weight decay for regularization
- Enabled `load_best_model_at_end`

### 5. Memory Management

**Changes**:
- Added GPU memory cleanup before and after each sweep run
- Prevents OOM errors when running multiple experiments

### 6. Hugging Face Hub Upload

**Changes**:
- Updated repository to `hayitsmaddy/mamamadal`
- Added `HF_TOKEN` authentication
- Updated all merge/save operations

---

## Final Configuration

### Hardware Requirements
- **GPU**: Tesla T4 (15GB VRAM) or better
- **RAM**: 13GB+ system RAM
- **Storage**: ~10GB for model + checkpoints

### Time Estimates (T4 GPU)
- **Per sweep run**: ~2-2.5 hours (1500 steps, 12k samples)
- **3 sweep runs**: ~6-8 hours total

### WandB Project Structure
- **Entity**: `hayleyc-kth-royal-institute-of-technology`
- **Project**: `uncategorized`
- **Sweep runs**: 3 experiments with different hyperparameters

---

## How to Use

### Google Colab
1. Add secrets: `hf_token`, `wandb_token`
2. Upload notebook
3. Enable T4 GPU
4. Run all cells

### Kaggle
1. Add secrets: `hf_token`, `wandb_token`
2. Upload notebook
3. Enable GPU T4 x2
4. Run all cells

### Local
1. Create `.env` file with `HF_TOKEN` and `WANDB_API_KEY`
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebook

---

## Key Differences from Original

| Aspect | Original | Modified |
|--------|----------|----------|
| Training | Single run | 3 sweep runs, Bayesian optimization |
| Dataset | Full 100k | 15% subsample (12k) |
| Training time | ~30 hours | ~2.5 hours per run |
| Overfitting | No prevention | Dropout, weight decay, early stopping |
| Environment | Colab only | Colab, Kaggle, Local |
| Metrics | train/loss only | train/loss + eval/loss |

---

**Last Updated**: 2025-11-30
