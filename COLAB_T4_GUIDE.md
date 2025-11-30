# Colab T4 Free Tier Optimization Guide

## Settings Applied

Your notebook is now optimized for **Google Colab T4 free tier** with these settings:

### Dataset Configuration
- **Subsampling**: 15% of FineTome-100k (~12,000 samples)
- **Purpose**: Fit multiple hyperparameter sweep runs within 12-hour Colab session
- **Toggle**: Set `USE_FULL_DATASET = True` for final training with best hyperparameters

### Training Arguments (Optimized for T4)
```python
per_device_train_batch_size = 2        # Max for T4 with 8B model
gradient_accumulation_steps = 4 or 8   # Effective batch size: 8 or 16
num_train_epochs = 1                   # Full epoch
save_steps = 500                       # Less frequent saves (save disk space)
save_total_limit = 2                   # Keep only 2 checkpoints
```

### Hyperparameter Sweep
```python
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'train/loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'min': 1e-5, 'max': 5e-4},
        'gradient_accumulation_steps': {'values': [4, 8]},
    }
}
# 3 sweep runs (fits in ~10-11 hours)
```

## Estimated Timing (Colab T4)
- **Per epoch with 12k samples**: ~3-4 hours
- **3 sweep runs**: ~10-11 hours total
- **Fits comfortably in 12-hour session limit** ✓

## Workflow
1. **Run sweeps** with 15% dataset (current setting)
2. **Review WandB** to find best hyperparameters
3. **Set** `USE_FULL_DATASET = True`
4. **Final training** with full 81k samples and best params

## Memory Usage (T4 - 15GB VRAM)
- Model (4-bit): ~5GB
- LoRA adapters: ~1GB
- Batch size 2: ~8GB
- **Total**: ~14GB ✓ (fits comfortably)

## Tips
- Monitor WandB for loss curves
- If session disconnects, checkpoints are saved to Google Drive
- Free Colab gives ~12 hours, but can disconnect earlier under load
