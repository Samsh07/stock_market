import matplotlib.pyplot as plt
import os

def plot_training_history(history, save_path='results/training_history.png'):
    """Plots training and validation loss/accuracy."""
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Training Loss', marker='o', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', marker='s', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history['val_acc'], label='Validation Accuracy', marker='o', 
                 linewidth=2, color='green')
    axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Set y-axis limits for accuracy plot
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def save_metrics(metrics, save_path='results/metrics.txt'):
    """Save evaluation metrics to a text file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("Model Evaluation Metrics\n")
        f.write("="*50 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Metrics saved to: {save_path}")