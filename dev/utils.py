import os
import sys
from pathlib import Path

def plot_training_history(history, report_folder):
    try:
        import matplotlib.pyplot as plt
        epochs = range(1, len(history['train_loss']) + 1)
        batch_epochs = range(1, len(history['batch_losses']) + 1)

        has_acc = 'train_acc' in history and 'val_acc' in history and \
                  len(history['train_acc']) == len(history['train_loss']) and \
                  len(history['val_acc']) == len(history['train_loss'])

        # --- Setup figure: 2 subplots if accuracy exists, else 1 ---
        plt.figure(figsize=(12, 5) if has_acc else (6, 5))

        # --- Loss Plot ---
        ax1 = plt.subplot(1, 2, 1) if has_acc else plt.subplot(1, 1, 1)
        ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')

        # Batch loss on secondary x-axis
        ax2 = ax1.twiny()
        # Scatterplot
        ax2.plot(batch_epochs, history['batch_losses'], 'go', markersize=1, alpha=0.15, label='Batch Losses')
        # Thin connecting line
        ax2.plot(batch_epochs, history['batch_losses'], 'g-', linewidth=0.1, alpha=0.1)        
        ax2.set_xlabel('Batch Steps')
        ax2.tick_params(axis='x', labelsize=8, colors='green')

        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax1.set_yscale('log')
        # --- Optional Accuracy Plot ---
        if has_acc:
            ax3 = plt.subplot(1, 2, 2)
            ax3.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
            ax3.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
            ax3.set_title('Training and Validation Accuracy')
            ax3.set_xlabel('Epochs')
            ax3.set_ylabel('Accuracy')
            ax3.grid(True)
            ax3.legend()

        plt.tight_layout()

        # --- Save plot ---
        fig_path = report_folder / 'training_history.png'
        os.makedirs(fig_path.parent, exist_ok=True)
        plt.savefig(fig_path)
        print(f"\nSaved training history plot to {fig_path.name}")
    except ImportError:
        print("\nMatplotlib not found. Skipping history plot.")
    except Exception as e:
        print(f"\nCould not generate plot. Error: {e}")
