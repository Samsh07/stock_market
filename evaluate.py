import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Import from correct modules
from sentiment_analyzer import FinancialNewsDataset, load_finetuned_model
from data_processing import load_data

def evaluate_model(model, dataloader, device):
    """Evaluates the model on a given dataloader."""
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy, predictions, true_labels

def plot_confusion_matrix(true_labels, predictions, save_path=None):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    # --- Configuration ---
    MODEL_PATH = '../models/finbert_sentiment_model'
    TEST_DATA_FILE = '../data/processed/test.csv'  # FIXED
    BATCH_SIZE = 8
    print("="*50)
    print("Starting Model Evaluation")
    print("="*50)
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Load Data ---
    print("\nLoading test data...")
    test_df = load_data(TEST_DATA_FILE)
    print(f"Test samples: {len(test_df)}")
    
    print("\nLoading fine-tuned model...")
    tokenizer, model = load_finetuned_model(MODEL_PATH)
    model.to(device)
    
    test_dataset = FinancialNewsDataset(test_df, tokenizer, max_token_length=256)  # Match training token length
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # --- Evaluate ---
    print("\nEvaluating model...")
    loss, accuracy, all_preds, all_labels = evaluate_model(model, test_dataloader, device)

    # --- Report ---
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    labels = ['Negative', 'Neutral', 'Positive']
    for i, label in enumerate(labels):
        print(f"\n{label}:")
        print(f"  Precision: {precision_per_class[i]:.4f}")
        print(f"  Recall: {recall_per_class[i]:.4f}")
        print(f"  F1-Score: {f1_per_class[i]:.4f}")
        print(f"  Support: {support[i]}")
    
    # Create results directory if it doesn't exist
    os.makedirs('../results', exist_ok=True)
    plot_confusion_matrix(all_labels, all_preds, save_path='../results/confusion_matrix.png')
    
    print("\n" + "="*50)
    print("Evaluation complete!")
    print("="*50)

if __name__ == '__main__':
    main()