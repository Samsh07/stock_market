import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import gc

# ===== GPU MEMORY OPTIMIZATION =====
def clear_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Enable memory efficient settings
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
# ===================================

from data_processing import load_data
from sentiment_analyzer import FinancialNewsDataset, load_model_from_local_path, save_model
from utils import plot_training_history
from evaluate import evaluate_model

def train_model(model, train_dataloader, val_dataloader, device, epochs=3, learning_rate=2e-5, gradient_accumulation_steps=2):
    """Trains the model and returns training history."""
    model.to(device)
    
    # Enable mixed precision training for memory efficiency
    scaler = GradScaler()
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
    total_steps = len(train_dataloader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        # Clear memory before each epoch
        clear_memory()
        
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Use mixed precision training
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps
            
            total_train_loss += loss.item() * gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
            
            # Clear memory every 100 batches
            if batch_idx % 100 == 0:
                clear_memory()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        clear_memory()
        val_loss, val_acc, _, _ = evaluate_model(model, val_dataloader, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)')
        
        # Print GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()/1e9
            reserved = torch.cuda.memory_reserved()/1e9
            print(f'  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved')
            torch.cuda.reset_peak_memory_stats()
    
    return history

def main():
    # --- Configuration ---
    TRAIN_FILE = '../data/processed/train.csv'
    VAL_FILE = '../data/processed/val.csv'
    TEST_FILE = '../data/processed/test.csv'
    
    MODEL_PATH = r"C:\Users\samsh\.cache\huggingface\hub\models--yiyanghkust--finbert-pretrain\snapshots\88ab954a39ea6d3ce2b62cff086dd5ad1172c664"
    SAVE_PATH = '../models/finbert_sentiment_model'
    MAX_TOKEN_LENGTH = 256  # Reduced from 512 to save memory
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 4  # Reduced from 8 to 4
    GRADIENT_ACCUMULATION_STEPS = 2  # Simulate batch size of 8
    
    # Create necessary directories
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    
    print("="*70)
    print(" "*15 + "FINANCIAL SENTIMENT ANALYSIS TRAINING")
    print("="*70)
    
    # --- Load Pre-processed Data ---
    print("\n[1/4] Loading pre-processed data...")
    train_df = load_data(TRAIN_FILE)
    val_df = load_data(VAL_FILE)
    test_df = load_data(TEST_FILE)
    
    print(f"  ✓ Train: {len(train_df):,} samples")
    print(f"  ✓ Val: {len(val_df):,} samples")
    print(f"  ✓ Test: {len(test_df):,} samples")
    print(f"  ✓ Total: {len(train_df) + len(val_df) + len(test_df):,} samples")
    
    # --- Model and Tokenizer ---
    print("\n[2/4] Loading model and tokenizer from local cache...")
    tokenizer, model = load_model_from_local_path(MODEL_PATH, num_labels=3)
    print(f"  ✓ Model loaded from local cache")
    
    # --- Datasets and DataLoaders ---
    print("\n[3/4] Creating datasets and dataloaders...")
    train_dataset = FinancialNewsDataset(train_df, tokenizer, MAX_TOKEN_LENGTH)
    val_dataset = FinancialNewsDataset(val_df, tokenizer, MAX_TOKEN_LENGTH)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"  ✓ Training batches: {len(train_dataloader):,}")
    print(f"  ✓ Validation batches: {len(val_dataloader):,}")
    print(f"  ✓ Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  ✓ Max token length: {MAX_TOKEN_LENGTH}")
    
    # --- Training ---
    print("\n[4/4] Starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  ✓ Using device: {device}")
    
    if device.type == 'cuda':
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  ✓ Mixed Precision Training: Enabled")
        print(f"  ✓ Memory Management: Enabled")
    
    print(f"\n  Training for {EPOCHS} epochs with learning rate {LEARNING_RATE}")
    print("="*70)
    
    try:
        history = train_model(
            model, 
            train_dataloader, 
            val_dataloader, 
            device, 
            epochs=EPOCHS, 
            learning_rate=LEARNING_RATE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
        )
        
        # --- Save Model ---
        print("\n" + "="*70)
        print("Saving trained model...")
        save_model(model, tokenizer, SAVE_PATH)
        print(f"  ✓ Model saved to: {SAVE_PATH}")
        
        # --- Plot History ---
        print("\nGenerating training history plot...")
        plot_training_history(history, save_path='../results/training_history.png')
        
        # --- Final Summary ---
        print("\n" + "="*70)
        print(" "*20 + "TRAINING COMPLETE!")
        print("="*70)
        print(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
        print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
        print(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f} ({history['val_acc'][-1]*100:.2f}%)")
        print(f"\nModel saved to: {SAVE_PATH}")
        print(f"Results saved to: ../results/")
        print("\nNext steps:")
        print("  1. Run 'python evaluate.py' to evaluate on test set")
        print("  2. Run 'python predict.py' to make predictions")
        print("="*70)
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n" + "!"*70)
            print("GPU OUT OF MEMORY ERROR!")
            print("!"*70)
            print("Try these solutions:")
            print("1. Reduce BATCH_SIZE to 2")
            print("2. Reduce MAX_TOKEN_LENGTH to 128")
            print("3. Use CPU: device = torch.device('cpu')")
            print("!"*70)
        raise

if __name__ == '__main__':
    main()