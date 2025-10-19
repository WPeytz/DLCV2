#!/bin/bash
#BSUB -q c02516
#BSUB -J Test
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 08:00
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err

set -euo pipefail

echo "=== Job started on $(hostname) at $(date) ==="
echo "Working directory: $(pwd)"

module load python3/3.9.19
module load cuda/11.8

# Activate your virtual environment
source ~/myproject/.venv/bin/activate || { echo "venv activation failed"; exit 1; }

# Go to your code folder (ABSOLUTE PATH!)
cd /zhome/e8/8/167932/DLCV2 || { echo "cd failed"; exit 1; }

echo "PWD: $(pwd)"
echo "Python: $(which python)"
python -V

# --- Global reproducibility knobs ---
export SEED=42
export PYTHONHASHSEED="$SEED"

# For PyTorch+CUDA determinism (safe for CUDA 11.8). No-op if PyTorch isn't used.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# (Optional) If you might use TensorFlow in some scripts:
export TF_DETERMINISTIC_OPS=1

# Install a sitecustomize hook into the active venv so ALL python invocations get seeded.
SITEPKG="$(python -c 'import site; print(next(p for p in (getattr(site, "getsitepackages", lambda: [])() or []) if p.endswith("site-packages")) or site.getusersitepackages())')"
mkdir -p "$SITEPKG"
cat > "$SITEPKG/sitecustomize.py" << 'PY'
import os, random
SEED = int(os.environ.get("SEED", "42"))

# Ensure Python's hash randomization is deterministic too (affects dict/set iteration order).
os.environ.setdefault("PYTHONHASHSEED", str(SEED))

# Standard library
random.seed(SEED)

# NumPy (if present)
try:
    import numpy as np
    np.random.seed(SEED)
except Exception:
    pass

# PyTorch (if present)
try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # cudnn determinism
    try:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
except Exception:
    pass

# TensorFlow (if present)
try:
    import tensorflow as tf
    try:
        tf.random.set_seed(SEED)
    except Exception:
        pass
except Exception:
    pass

# Optional: print once per process so you can verify it worked
if os.environ.get("SEED_VERBOSE", "1") == "1":
    print(f"[sitecustomize] Global seed set to {SEED}")
PY


# Run your training script with unbuffered output so print()s show up immediately
python train.py --model dualstream --fusion mean --root_dir ucf101_noleakage --epochs 100
#python train.py --model 3dcnn --batch_size 8 --epochs 2


echo "=== Job finished at $(date) ==="
