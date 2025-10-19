"""
Utility functions for training and evaluation.
Now includes seeding helpers for reproducibility.
"""

import os
import shutil
import random
from contextlib import contextmanager
from typing import Optional, Dict, Any

import numpy as np
import torch


# ---------------------- Reproducibility helpers ---------------------- #
def set_global_seed(seed: int, deterministic: bool = True) -> torch.Generator:
    """
    Set global RNG seeds for Python, NumPy, and PyTorch (CPU + CUDA).
    Optionally enable deterministic algorithms. Returns a torch.Generator
    suitable for DataLoader's 'generator' argument (deterministic shuffling).

    Example:
        g = set_global_seed(42, deterministic=True)
        DataLoader(..., generator=g, worker_init_fn=make_worker_init_fn(42))
    """
    seed = int(seed)

    # Python & NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch (CPU + CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Disable cuDNN autotuner; enable deterministic kernels where possible
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older torch versions may not support this; safe to ignore.
            pass

        # Encourage deterministic cublas behavior on supported ops
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # DataLoader shuffling generator
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def make_worker_init_fn(base_seed: int):
    """
    Create a worker_init_fn for DataLoader that seeds Python/NumPy in each worker
    based on the worker's initial torch seed (derived from base_seed). This makes
    per-worker randomness reproducible but distinct across workers.

    Usage:
        worker_init_fn = make_worker_init_fn(42)
        DataLoader(..., worker_init_fn=worker_init_fn)
    """
    base_seed = int(base_seed)

    def _init_fn(worker_id: int):
        worker_seed = (torch.initial_seed() + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return _init_fn


@contextmanager
def seed_context(seed: Optional[int]):
    """
    Temporarily fork torch RNG (CPU and CUDA) and, if seed is provided, seed it.
    Useful for deterministic parameter initialization that doesn't disturb
    the caller's global RNG state.

    Example:
        with seed_context(123):
            layer.reset_parameters()
    """
    if seed is None:
        yield
        return

    with torch.random.fork_rng(enabled=True):
        torch.manual_seed(int(seed))
        try:
            torch.cuda.manual_seed_all(int(seed))
        except Exception:
            pass
        yield


def get_rng_state_dict() -> Dict[str, Any]:
    """
    Capture current RNG states so you can store them in a checkpoint for
    bit-exact resumption.

    Returns:
        dict with keys: 'python', 'numpy', 'torch_cpu', 'torch_cuda'
    """
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            # Fallback if multi-GPU state isn't available
            state["torch_cuda"] = torch.cuda.get_rng_state()
    else:
        state["torch_cuda"] = None
    return state


def set_rng_state_dict(state: Dict[str, Any]) -> None:
    """
    Restore RNG states captured by get_rng_state_dict(). Safe to call even if
    some keys are missing or CUDA isn't available.
    """
    if not isinstance(state, dict):
        return

    try:
        if "python" in state and state["python"] is not None:
            random.setstate(state["python"])
    except Exception:
        pass

    try:
        if "numpy" in state and state["numpy"] is not None:
            np.random.set_state(state["numpy"])
    except Exception:
        pass

    try:
        if "torch_cpu" in state and state["torch_cpu"] is not None:
            torch.set_rng_state(state["torch_cpu"])
    except Exception:
        pass

    try:
        if torch.cuda.is_available() and "torch_cuda" in state and state["torch_cuda"] is not None:
            if isinstance(state["torch_cuda"], (list, tuple)):
                torch.cuda.set_rng_state_all(state["torch_cuda"])
            else:
                torch.cuda.set_rng_state(state["torch_cuda"])
    except Exception:
        pass
# -------------------------------------------------------------------- #


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output: Model predictions [batch_size, num_classes]
        target: Ground truth labels [batch_size]
        topk: Tuple of k values

    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, checkpoint_path, checkpoint_dir, model_name):
    """
    Save model checkpoint.

    Args:
        state: Dictionary containing model state, optimizer state, etc.
               (Tip: you can add RNG for exact resume:
                    state['rng_state'] = get_rng_state_dict())
        is_best: Boolean indicating if this is the best model so far
        checkpoint_path: Path to save the checkpoint
        checkpoint_dir: Directory to save checkpoints
        model_name: Name of the model
    """
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
        shutil.copyfile(checkpoint_path, best_path)


def load_checkpoint(checkpoint_path):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary containing checkpoint data
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    """
    Get current learning rate from optimizer.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


class EarlyStopping:
    """
    Early stopping to stop training when validation performance stops improving.
    """

    def __init__(self, patience=10, min_delta=0, mode='max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics that should increase (accuracy),
                  'min' for metrics that should decrease (loss)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


def freeze_layers(model, freeze_until=None):
    """
    Freeze model layers up to a certain point.

    Args:
        model: PyTorch model
        freeze_until: Layer name to freeze until (inclusive)
    """
    freeze = True
    for name, param in model.named_parameters():
        if freeze:
            param.requires_grad = False
        if freeze_until is not None and freeze_until in name:
            freeze = False


def unfreeze_all(model):
    """
    Unfreeze all model parameters.

    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True


if __name__ == "__main__":
    # Quick self-test of utilities
    import torch.nn as nn

    # Reproducibility smoke test
    g = set_global_seed(42, deterministic=True)
    w1 = torch.randn(3)
    with seed_context(123):
        w2 = torch.randn(3)
    w3 = torch.randn(3)
    assert not torch.allclose(w1, w2), "seed_context should isolate RNG"
    assert not torch.allclose(w1, w3), "Subsequent draws should advance RNG"

    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg}")

    # Test accuracy
    output = torch.randn(4, 10, generator=g)
    target = torch.randint(0, 10, (4,), generator=g)
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    print(f"Top-1 Accuracy: {acc1.item():.2f}%")
    print(f"Top-5 Accuracy: {acc5.item():.2f}%")

    # Test parameter counting
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    print(f"Model parameters: {count_parameters(model)}")
