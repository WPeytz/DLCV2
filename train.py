"""
Training script for video classification models.
Supports all model types from Project 4.1 and 4.2.
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import json

from datasets import FrameImageDataset, FrameVideoDataset
from models import (PerFrameCNN, LateFusionCNN, EarlyFusionCNN, CNN3D,
                    PerFrame3D, LateFusion3D, DualStreamNetwork)
from utils import AverageMeter, accuracy, save_checkpoint, load_checkpoint


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
                        choices=['perframe', 'latefusion', 'earlyfusion', '3dcnn',
                                 'perframe3d', 'latefusion3d', 'dualstream'],
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
    parser.add_argument('--clip_size', type=int, default=3,
                        help='Clip size for PerFrame3D model')
    parser.add_argument('--num_segments', type=int, default=3,
                        help='Number of segments for LateFusion3D model')

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

    # Other parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate model')
    parser.add_argument('--eval_split', type=str, default='val',
                        choices=['val', 'test'],
                        help='Split to evaluate when using --eval_only')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    elif args.model == 'perframe3d':
        model = PerFrame3D(
            num_classes=num_classes,
            clip_size=args.clip_size,
            aggregation=args.aggregation
        )
    elif args.model == 'latefusion3d':
        model = LateFusion3D(
            num_classes=num_classes,
            num_segments=args.num_segments,
            fusion=args.fusion
        )
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


def get_dataloaders(args):
    """Create train, validation, and test dataloaders."""

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
        # Use frame dataset for per-frame 2D models
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
        test_dataset = FrameImageDataset(
            root_dir=args.root_dir,
            split='test',
            transform=val_transform
        )
    else:
        # Use video dataset for all other models (3D CNNs, fusion models, dual-stream)
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
        test_dataset = FrameVideoDataset(
            root_dir=args.root_dir,
            split='test',
            transform=val_transform,
            stack_frames=True,
            load_optical_flow=load_flow
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


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
            rgb = data['frames'].to(device)
            flow = data['flow'].to(device)
            labels = data['label'].to(device)

            outputs = model(rgb, flow)
        else:
            # Other models
            if isinstance(batch, dict):
                inputs = batch['frames'].to(device)
                labels = batch['label'].to(device)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

            outputs = model(inputs)

        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
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
                rgb = data['frames'].to(device)
                flow = data['flow'].to(device)
                labels = data['label'].to(device)

                outputs = model(rgb, flow)
            else:
                if isinstance(batch, dict):
                    inputs = batch['frames'].to(device)
                    labels = batch['label'].to(device)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

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

    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

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
    train_loader, val_loader, test_loader = get_dataloaders(args)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

    # Evaluation only
    if args.eval_only:
        eval_loader = test_loader if args.eval_split == 'test' else val_loader
        eval_loss, eval_acc = validate(model, eval_loader, criterion, device, args)
        print(f"{args.eval_split.capitalize()} Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.2f}%")
        return

    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

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

    # Save training history
    history_path = os.path.join(args.checkpoint_dir, f'{args.model}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)

    print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")

    # Test evaluation with best model
    print("\n" + "="*50)
    print("Evaluating on test set with best model...")
    print("="*50)

    # Load best checkpoint
    best_checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.model}_best.pth')
    if os.path.exists(best_checkpoint_path):
        print(f"Loading best checkpoint from {best_checkpoint_path}")
        checkpoint = load_checkpoint(best_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best checkpoint from epoch {checkpoint['epoch']} with val acc: {checkpoint['best_acc']:.2f}%")
    else:
        print("Warning: Best checkpoint not found, using final model")

    # Evaluate on test set
    test_loss, test_acc = validate(model, test_loader, criterion, device, args)
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # Save test results
    results = {
        'best_val_acc': best_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'model': args.model,
        'backbone': args.backbone if hasattr(args, 'backbone') else None,
        'num_frames': args.num_frames,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr
    }

    results_path = os.path.join(args.checkpoint_dir, f'{args.model}_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nTest results saved to: {results_path}")
    print(f"Final Results - Val Acc: {best_acc:.2f}% | Test Acc: {test_acc:.2f}%")


if __name__ == '__main__':
    main()
