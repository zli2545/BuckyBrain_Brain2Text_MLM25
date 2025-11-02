# BuckyBrain Neural Decoder

Neural activity to phoneme decoding using GRU with CTC loss.

## Features

- **Modular design**: Clean separation of dataset, model, trainer, and utilities
- **Flexible session management**: Auto-discovery or manual specification of recording sessions
- **Day-specific calibration**: Per-session input layers for cross-day generalization
- **Comprehensive logging**: File and console logging with per-day validation metrics
- **Parameter grouping**: Separate learning rates and regularization for different parameter types
- **Data augmentation**: Noise injection, temporal jitter, Gaussian smoothing
- **Mixed precision**: Automatic mixed precision (AMP) for faster training
- **Configuration-driven**: YAML config for easy experimentation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Debug Mode (Fast Validation)

**For GPU users:**
```bash
python scripts/train_neural_decoder.py --debug --gpu 0
```
- Uses 100 training batches (~5 minutes on GPU)
- Validates every 50 batches
- Outputs to `./outputs_debug/`

**For CPU users (team members testing architectures):**
```bash
python scripts/train_neural_decoder.py --debug --debug_batches 20 --gpu -1
```
- Uses 20 training batches (~30-60 minutes on CPU)
- Validates every 10 batches
- Outputs to `./outputs_debug/`
- See [Testing and Debugging](#testing-and-debugging-cpu-workflow) section for details

### 2. Full Training

Edit `configs/neural_decoder.yaml` to configure training, then:

```bash
python scripts/train_neural_decoder.py --config configs/neural_decoder.yaml --gpu 0
```

Or run with nohup for long training:

```bash
nohup python scripts/train_neural_decoder.py --config configs/neural_decoder.yaml --gpu 0 > train.log 2>&1 &
```

### 3. Evaluation

```bash
python scripts/eval_phonemes.py \
    --ckpt outputs/best.pt \
    --dataset_dir /path/to/hdf5_data_final \
    --split val \
    --gpu 0
```

## Configuration

Key configuration options in `configs/neural_decoder.yaml`:

### Session Management

```yaml
dataset:
  # Auto-discover all available sessions
  sessions_train: null
  sessions_val: null
  auto_discover: true
  
  # OR manually specify for debugging
  sessions_train: ['t15.2023.08.11', 't15.2023.08.13']
  sessions_val: ['t15.2023.08.11']
  auto_discover: false
```

### Model Architecture

```yaml
model:
  n_units: 768        # GRU hidden units
  n_layers: 5         # Number of GRU layers
  patch_size: 14      # Temporal downsampling
  patch_stride: 4
```

### Training

```yaml
training:
  batch_size: 64
  days_per_batch: 4   # Mix data from N days per batch
  n_train_batches: 120000
  val_every: 2000
```

### Optimizer

```yaml
optimizer:
  lr: 0.005           # Base learning rate
  lr_day: 0.005       # LR for day-specific layers
  weight_decay: 0.001 # L2 regularization
  weight_decay_day: 0.0
  grad_clip: 10.0
```

## Project Structure

```
BuckyBrain_Brain2Text_MLM25/
├── configs/                    # Configuration files
│   ├── neural_decoder.yaml    # Neural decoder config
│   ├── language_model.yaml    # Language model config (future)
│   └── end_to_end.yaml        # End-to-end pipeline config (future)
│
├── nn_model/                   # Neural decoder module (activity → phoneme)
│   ├── __init__.py
│   ├── dataset.py             # Data loading and batching
│   ├── model.py               # GRU decoder architecture
│   ├── trainer.py             # Training loop and validation
│   └── utils.py               # Augmentation and utilities
│
├── lm_model/                   # Language model module (phoneme → text, future)
│   └── __init__.py            # Placeholder for future LM components
│
└── scripts/                    # Executable scripts
    ├── train_neural_decoder.py   # Train neural decoder
    ├── eval_phonemes.py          # Evaluate phoneme-level decoding
    ├── train_language_model.py   # Train LM (future)
    ├── eval_text.py              # Evaluate text-level (future)
    └── inference.py              # Real-time inference (future)
```

## Expected Performance

With default settings (120k batches):
- Validation PER: ~10% (comparable to original paper)
- Training time: ~3-4 hours on RTX 4090

For debugging (500 batches):
- PER will be high (~50-80%) but loss should decrease
- Training time: ~5 minutes

## Testing and Debugging (CPU Workflow)

**For team members experimenting with model architectures: Use CPU for quick validation, then report back for GPU training.**

### Modifying Model Architecture

To experiment with your own architecture:

1. **Edit the model** in `nn_model/model.py`:
   ```python
   # Modify GRUDecoder class in nn_model/model.py
   class GRUDecoder(nn.Module):
       def __init__(self, ...):
           # Your architecture changes here
           self.your_custom_layer = nn.Linear(...)
       
       def forward(self, x, day_idx, ...):
           # Your forward pass modifications
           x = self.your_custom_layer(x)
           ...
   ```

2. **Update config** if needed:
   ```yaml
   model:
     n_units: 512  # Change if your architecture requires it
     n_layers: 3   # Modify as needed
   ```

3. **Test on CPU** (see below)

### CPU Testing Workflow

The recommended workflow for testing new model architectures:

1. **Experiment on your branch**: Make architecture changes in `nn_model/model.py`
2. **Quick CPU test**: Run minimal debug mode to verify code works
3. **Report results**: If CPU test passes, report to main developer for GPU training

### CPU Debug Mode (Recommended for Testing)

**Minimal test** (20 batches, ~30-60 minutes on CPU):
```bash
python scripts/train_neural_decoder.py \
    --debug \
    --debug_batches 20 \
    --gpu -1
```

**Standard debug** (100 batches, ~3-6 hours on CPU):
```bash
python scripts/train_neural_decoder.py \
    --debug \
    --gpu -1
```

**What debug mode does:**
- ✅ Uses your config file (preserves your architecture settings)
- ✅ Reduces batch count (default: 100, can customize)
- ✅ Outputs to `./outputs_debug/` to avoid overwriting main outputs
- ✅ Validates frequently to catch errors early

### CPU Testing Configuration

For CPU testing, the code automatically:
- ✅ Disables AMP (CPU doesn't support bfloat16)
- ✅ Uses float32 precision
- ✅ Adjusts autocast settings

**You only need to set `gpu: -1`** (or use `--gpu -1` flag):

```bash
# Option 1: Command-line flag (recommended)
python scripts/train_neural_decoder.py --debug --gpu -1

# Option 2: Edit config (not required)
# system:
#   gpu: -1
```

### Expected CPU Performance

| Test Scope | Batches | CPU Time | Purpose |
|-----------|---------|----------|---------|
| **Minimal** | 20 | 30-60 min | ✅ Quick syntax/import check |
| **Basic** | 50 | 1.5-3 hrs | ✅ Verify data loading works |
| **Standard** | 100 | 3-6 hrs | ✅ Check loss decreases |
| **Extended** | 500 | 15-30 hrs | ⚠️ Not recommended on CPU |

### What to Verify in CPU Test

After running debug mode, check:

1. **Code runs without errors**:
   ```
   ✓ No import errors
   ✓ No syntax errors
   ✓ Data loads successfully
   ```

2. **Training starts**:
   ```
   ✓ Model initializes
   ✓ Loss is computed (should start ~3.5-4.0)
   ✓ Gradient is computed
   ```

3. **Loss decreases** (even slightly):
   ```
   Batch 10:  loss: 3.8
   Batch 50:  loss: 3.6  ← Should decrease
   Batch 100: loss: 3.4  ← Should decrease
   ```

4. **No crashes**:
   ```
   ✓ No OOM errors
   ✓ No NaN losses
   ✓ No gradient explosion (grad_norm < 1000)
   ```

### CPU Testing Checklist

Before reporting your architecture changes:

- [ ] Code runs without errors on CPU
- [ ] Loss decreases (even if high)
- [ ] No NaN or infinite values
- [ ] Checkpoint saves successfully (`outputs_debug/best.pt`)
- [ ] Validation PER is computed (even if high, e.g., 70-90%)
- [ ] Training log shows expected metrics

### CPU vs GPU Performance

| Metric | CPU | GPU | Ratio |
|-------|-----|-----|-------|
| **Training speed** | ~10-30 batch/hr | ~600 batch/hr | 20-60x slower |
| **Debug mode (100 batches)** | 3-6 hours | 5-10 minutes | 30-40x slower |
| **Full training (120k batches)** | 170-330 hours | 3-4 hours | 50-100x slower |

**Recommendation**: Use CPU only for validation, not for full training.

### Common Issues in CPU Testing

**Issue**: Training very slow
- **Cause**: Normal for CPU (20-60x slower than GPU)
- **Solution**: This is expected. Use small `--debug_batches` for quick tests

**Issue**: Out of memory
- **Solution**: Reduce `batch_size: 8` in config

**Issue**: Warning about autocast
- **Cause**: CPU doesn't support bfloat16
- **Solution**: Already handled automatically, can ignore

### After CPU Test Passes

If your CPU test shows:
- ✅ Code runs without errors
- ✅ Loss decreases
- ✅ No crashes

**Report back with:**
1. Your branch name
2. Architecture changes made
3. CPU test results (loss values, any warnings)
4. Config file used

Then the main developer will run full GPU training for performance evaluation.

## Tips

1. **Start small**: Use debug mode first to verify data loading works
2. **Monitor per-day PER**: Some days may need more training
3. **Adjust days_per_batch**: Higher = more day mixing, lower = faster convergence per day
4. **GPU memory**: Reduce `batch_size` if OOM occurs
5. **Data augmentation**: Tune noise levels if validation PER plateaus

## Data Format

Expects HDF5 files in this structure:

```
hdf5_data_final/
├── t15.2023.08.11/
│   ├── data_train.hdf5
│   └── data_val.hdf5
├── t15.2023.08.13/
│   ├── data_train.hdf5
│   └── data_val.hdf5
...
```

Each HDF5 file contains:
- `trial_0000`, `trial_0001`, ... groups
- Each trial has:
  - `input_features`: Neural data [time, features]
  - `seq_class_ids`: Phoneme labels [seq_len]
  - Attributes: `n_time_steps`, `seq_len`, `block_num`, `trial_num`

## Troubleshooting

**OOM Error**: Reduce `batch_size` or `n_units`

**Slow training (CPU)**: 
- This is expected! CPU is 20-60x slower than GPU
- Use `--debug_batches 20` for minimal testing
- See [CPU Testing section](#testing-and-debugging-cpu-workflow) for details

**Slow training (GPU)**: Enable `use_compile: true` (requires PyTorch 2.0+)

**High PER**: 
- Increase `n_train_batches`
- Tune data augmentation
- Check per-day PER for outliers

**Import errors**: Make sure you're in the project root directory

**CPU autocast warning**: 
- This is normal and can be ignored
- Code automatically handles CPU limitations
- Training will continue normally

## Quick Reference for Team Members

### Testing Your Architecture Changes on CPU

**Step 1: Make your changes**
```bash
# Create a branch
git checkout -b your-feature-branch

# Edit model architecture
vim nn_model/model.py  # or your IDE
```

**Step 2: Quick CPU test (20 batches, ~30-60 min)**
```bash
python scripts/train_neural_decoder.py \
    --debug \
    --debug_batches 20 \
    --gpu -1
```

**Step 3: Check results**
```bash
# Check if training completed
tail outputs_debug/training.log

# Look for:
# ✓ Loss decreases (e.g., 3.8 → 3.6)
# ✓ No errors
# ✓ Checkpoint saved (outputs_debug/best.pt)
```

**Step 4: Report back**
If test passes, share:
- Branch name
- Architecture changes
- CPU test results (loss values)
- Config file used

Then main developer will run full GPU training.

### Common CPU Testing Commands

```bash
# Minimal test (20 batches)
python scripts/train_neural_decoder.py --debug --debug_batches 20 --gpu -1

# Standard test (100 batches)  
python scripts/train_neural_decoder.py --debug --gpu -1

# Custom batch count
python scripts/train_neural_decoder.py --debug --debug_batches 50 --gpu -1

# Check training progress
tail -f outputs_debug/training.log
```

## License

MIT License - see original nejm-brain-to-text repository for reference implementation.
