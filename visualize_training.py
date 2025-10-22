"""
Visualize training results from HPC output files.
"""

import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_training_log(log_file):
    """Parse training log file and extract metrics."""
    with open(log_file, 'r') as f:
        content = f.read()

    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    learning_rates = []

    # Pattern to match training output
    pattern = r'Epoch (\d+)/\d+\nTrain Loss: ([\d.]+), Train Acc: ([\d.]+)%\nVal Loss: ([\d.]+), Val Acc: ([\d.]+)%\nLearning Rate: ([\d.]+)'

    matches = re.findall(pattern, content)

    for match in matches:
        epoch, train_loss, train_acc, val_loss, val_acc, lr = match
        epochs.append(int(epoch))
        train_losses.append(float(train_loss))
        train_accs.append(float(train_acc))
        val_losses.append(float(val_loss))
        val_accs.append(float(val_acc))
        learning_rates.append(float(lr))

    # Extract model info and best accuracy
    model_match = re.search(r'Model: (\w+)', content)
    model_name = model_match.group(1) if model_match else "Unknown"

    best_acc_match = re.search(r'Best validation accuracy: ([\d.]+)%', content)
    best_acc = float(best_acc_match.group(1)) if best_acc_match else None

    return {
        'epochs': epochs,
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs,
        'learning_rate': learning_rates,
        'model_name': model_name,
        'best_acc': best_acc
    }


def plot_training_curves(data, save_path=None):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = data['epochs']

    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, data['train_loss'], label='Train Loss', linewidth=2, color='#2E86AB')
    ax1.plot(epochs, data['val_loss'], label='Val Loss', linewidth=2, color='#A23B72')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, data['train_acc'], label='Train Acc', linewidth=2, color='#2E86AB')
    ax2.plot(epochs, data['val_acc'], label='Val Acc', linewidth=2, color='#A23B72')
    if data['best_acc']:
        best_epoch = epochs[np.argmax(data['val_acc'])]
        ax2.axhline(y=data['best_acc'], color='red', linestyle='--',
                    label=f'Best Val Acc: {data["best_acc"]:.2f}%', linewidth=1.5)
        ax2.scatter([best_epoch], [data['best_acc']], color='red', s=100, zorder=5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Learning rate schedule
    ax3 = axes[1, 0]
    ax3.plot(epochs, data['learning_rate'], linewidth=2, color='#F18F01')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Overfitting analysis (Train-Val gap)
    ax4 = axes[1, 1]
    acc_gap = np.array(data['train_acc']) - np.array(data['val_acc'])
    loss_gap = np.array(data['val_loss']) - np.array(data['train_loss'])

    ax4_twin = ax4.twinx()
    line1 = ax4.plot(epochs, acc_gap, label='Acc Gap (Train - Val)',
                     linewidth=2, color='#E63946')
    line2 = ax4_twin.plot(epochs, loss_gap, label='Loss Gap (Val - Train)',
                          linewidth=2, color='#06A77D', linestyle='--')

    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy Gap (%)', fontsize=12, color='#E63946')
    ax4_twin.set_ylabel('Loss Gap', fontsize=12, color='#06A77D')
    ax4.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='#E63946')
    ax4_twin.tick_params(axis='y', labelcolor='#06A77D')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, fontsize=11, loc='upper left')
    ax4.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'{data["model_name"].upper()} Training Results',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def print_summary(data):
    """Print training summary statistics."""
    print("\n" + "="*60)
    print(f"TRAINING SUMMARY - {data['model_name'].upper()}")
    print("="*60)
    print(f"Total Epochs: {len(data['epochs'])}")
    print(f"Best Validation Accuracy: {data['best_acc']:.2f}%")

    best_epoch = data['epochs'][np.argmax(data['val_acc'])]
    print(f"Best Epoch: {best_epoch}")

    final_train_acc = data['train_acc'][-1]
    final_val_acc = data['val_acc'][-1]
    print(f"Final Train Accuracy: {final_train_acc:.2f}%")
    print(f"Final Val Accuracy: {final_val_acc:.2f}%")
    print(f"Final Overfitting Gap: {final_train_acc - final_val_acc:.2f}%")

    print(f"\nInitial Learning Rate: {data['learning_rate'][0]:.6f}")
    print(f"Final Learning Rate: {data['learning_rate'][-1]:.6f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualize_training.py <output_file.out> [save_path.png]")
        print("\nExample:")
        print("  python visualize_training.py Output_26563948.out")
        print("  python visualize_training.py Output_26563948.out results.png")
        sys.exit(1)

    log_file = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Parse the log file
    print(f"Parsing log file: {log_file}")
    data = parse_training_log(log_file)

    # Print summary
    print_summary(data)

    # Plot curves
    plot_training_curves(data, save_path)
