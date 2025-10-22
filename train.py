"""
Training script for video classification models.
Supports all model types from Project 4.1 and 4.2.
Now includes deterministic seeding across Python, NumPy, PyTorch, CUDA, and DataLoader workers.
"""

import argparse
import os
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from datasets import FrameImageDataset, FrameVideoDataset
from models import PerFrameCNN, LateFusionCNN, EarlyFusionCNN, CNN3D, DualStreamNetwork
from utils import AverageMeter, accuracy, save_checkpoint, load_checkpoint


# ---------------------- Seeding utilities ---------------------- #
def set_global_seed(seed: int, deterministic: bool = True) -> torch.Generator:
    """
    Set seeds for Python, NumPy, and PyTorch. Optionally enable deterministic
    algorithms in PyTorch/CUDA. Returns a torch.Generator you can feed to
    DataLoader (for reproducible shuffling).
    """
    # For Python hashing and stdlib random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU + CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cudnn / deterministic kernels
    if deterministic:
        # Disables convolution autotuner for determinism
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older PyTorch versions won't have this; safe to ignore.
            pass

        # Helps cuBLAS use deterministic paths on some ops
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Generator for DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def make_worker_init_fn(base_seed: int):
    """
    Returns a worker_init_fn that seeds Python/NumPy for each worker process
    based on the worker's initial torch seed (derived from base_seed), so
    workers are deterministic but not identical to each other.
    """
    def _init_fn(worker_id: int):
        worker_seed = (torch.initial_seed() + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return _init_fn
# --------------------------------------------------------------- #

#   Dummy tqdm for silent epochs

class DummyTqdm:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable
    
    def __iter__(self):
        return iter(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
        
    def set_postfix(self, *args, **kwargs):
        # Essential fix: does nothing
        pass

#   --------------------------------------------------------------- #


def get_args():
    parser = argparse.ArgumentParser(description='Train video classification model')

    # Data parameters
    parser.add_argument('--root_dir', type=str, default='data/ufc10',
                        help='Root directory of dataset')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num_frames', type=int, default=10,
                        help='Number of frames per video')

    # Model parameters
    parser.add_argument('--model', type=str, default='perframe',
                        choices=['perframe', 'latefusion', 'earlyfusion', '3dcnn', 'dualstream'],
                        help='Model architecture')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'],
                        help='Backbone architecture (for CNN models)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--aggregation', type=str, default='mean',
                        choices=['mean', 'max'],
                        help='Aggregation method for per-frame model')
    parser.add_argument('--fusion', type=str, default='mean',
                        choices=['mean', 'concat', 'late'],
                        help='Fusion method for fusion models')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Reproducibility parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Global random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic algorithms (may reduce speed)')

    # Other parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate model')

    return parser.parse_args()


def get_model(args):
    """Create model based on arguments."""
    num_classes = 10

    if args.model == 'perframe':
        model = PerFrameCNN(
            num_classes=num_classes,
            backbone=args.backbone,
            pretrained=args.pretrained,
            aggregation=args.aggregation
        )
    elif args.model == 'latefusion':
        model = LateFusionCNN(
            num_classes=num_classes,
            backbone=args.backbone,
            pretrained=args.pretrained,
            fusion=args.fusion
        )
    elif args.model == 'earlyfusion':
        model = EarlyFusionCNN(
            num_classes=num_classes,
            backbone=args.backbone,
            pretrained=args.pretrained,
            num_frames=args.num_frames
        )
    elif args.model == '3dcnn':
        model = CNN3D(num_classes=num_classes)
    elif args.model == 'dualstream':
        model = DualStreamNetwork(
            num_classes=num_classes,
            backbone=args.backbone,
            pretrained=args.pretrained,
            fusion=args.fusion
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model


def get_dataloaders(args, dl_generator: torch.Generator):
    """Create train and validation dataloaders."""

    # Data transforms
    train_transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Choose dataset based on model type
    if args.model == 'perframe':
        # Use frame dataset for per-frame models
        train_dataset = FrameImageDataset(
            root_dir=args.root_dir,
            split='train',
            transform=train_transform
        )
        val_dataset = FrameImageDataset(
            root_dir=args.root_dir,
            split='val',
            transform=val_transform
        )
    else:
        # Use video dataset for other models
        load_flow = (args.model == 'dualstream')

        train_dataset = FrameVideoDataset(
            root_dir=args.root_dir,
            split='train',
            transform=train_transform,
            stack_frames=True,
            load_optical_flow=load_flow
        )
        val_dataset = FrameVideoDataset(
            root_dir=args.root_dir,
            split='val',
            transform=val_transform,
            stack_frames=True,
            load_optical_flow=load_flow
        )

    worker_init_fn = make_worker_init_fn(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=dl_generator,            # deterministic shuffling
        worker_init_fn=worker_init_fn,     # per-worker seeding
        persistent_workers=args.num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,     # keeps NumPy/random deterministic in workers
        persistent_workers=args.num_workers > 0
    )

    return train_loader, val_loader


def train_epoch(model, dataloader, criterion, optimizer, device, args):
    """Train for one epoch."""
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    pbar = tqdm(dataloader, desc='Training')

    for batch in pbar:
        if args.model == 'dualstream':
            # Dual-stream model needs RGB and flow
            data = batch
            rgb = data['frames'].to(device, non_blocking=True)
            flow = data['flow'].to(device, non_blocking=True)
            labels = data['label'].to(device, non_blocking=True)

            outputs = model(rgb, flow)
        else:
            # Other models
            if isinstance(batch, dict):
                inputs = batch['frames'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
            else:
                inputs, labels = batch
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)

        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Measure accuracy and record loss
        acc1 = accuracy(outputs, labels, topk=(1,))[0]
        losses.update(loss.item(), labels.size(0))
        top1.update(acc1.item(), labels.size(0))

        # Update progress bar
        pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'acc': f'{top1.avg:.2f}'})

    return losses.avg, top1.avg


def validate(model, dataloader, criterion, device, args):
    """Validate model."""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')

        for batch in pbar:
            if args.model == 'dualstream':
                data = batch
                rgb = data['frames'].to(device, non_blocking=True)
                flow = data['flow'].to(device, non_blocking=True)
                labels = data['label'].to(device, non_blocking=True)

                outputs = model(rgb, flow)
            else:
                if isinstance(batch, dict):
                    inputs = batch['frames'].to(device, non_blocking=True)
                    labels = batch['label'].to(device, non_blocking=True)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)

            loss = criterion(outputs, labels)

            # Measure accuracy and record loss
            acc1 = accuracy(outputs, labels, topk=(1,))[0]
            losses.update(loss.item(), labels.size(0))
            top1.update(acc1.item(), labels.size(0))

            pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'acc': f'{top1.avg:.2f}'})

    return losses.avg, top1.avg


def main():
    args = get_args()

    # Set global seed & determinism; get a DL generator for shuffling
    dl_generator = set_global_seed(args.seed, deterministic=args.deterministic)
    print(f"[seed] Using seed={args.seed} deterministic={bool(args.deterministic)}")

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Model
    model = get_model(args)
    model = model.to(device)
    print(f"Model: {args.model}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print(f"Resumed from epoch {start_epoch}")

    # Data loaders
    train_loader, val_loader = get_dataloaders(args, dl_generator)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # Evaluation only
    if args.eval_only:
        val_loss, val_acc = validate(model, val_loader, criterion, device, args)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        return

    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    global tqdm
    original_tqdm = tqdm  # Save original tqdm for later use

    for epoch in range(start_epoch, args.epochs):
        is_print_epoch = (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs

        if is_print_epoch:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            tqdm = original_tqdm
        else:
            # Disable progress bar for silent epochs
            tqdm = DummyTqdm

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, args)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, args)

        # Update learning rate
        scheduler.step(val_acc)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print epoch results
        if is_print_epoch:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.model}_latest.pth')
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'args': vars(args)
        }, is_best, checkpoint_path, args.checkpoint_dir, args.model)

        # Reset tqdm state for the next check
        tqdm = original_tqdm

    # Save training history
    history_path = os.path.join(args.checkpoint_dir, f'{args.model}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)

    print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
