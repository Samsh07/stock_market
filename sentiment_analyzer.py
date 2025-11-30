import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import re
import os

class FinancialNewsDataset(Dataset):
    """Custom Dataset for financial news."""
    def __init__(self, dataframe, tokenizer, max_token_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.Content.values
        self.targets = dataframe.Sentiment.values
        self.max_token_length = max_token_length
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        text = str(self.text[index])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        target = torch.tensor(self.targets[index], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': target
        }

def load_model(model_name='yiyanghkust/finbert-tone', num_labels=3):
    """Loads a pre-trained model and tokenizer."""
    
    # FIXED: Use the actual cache directory where your model is downloaded
    # Option 1: Let Hugging Face use default cache (RECOMMENDED)
    # This will automatically find your downloaded model
    print(f"Loading model: {model_name}")
    print("Using Hugging Face default cache directory")
    
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        print("Model loaded successfully from cache!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying alternative model names...")
        
        # Try alternative model names in case of typo
        alternative_models = [
            'yiyanghkust/finbert-pretrain',  # The one you actually downloaded
            'ProsusAI/finbert'
        ]
        
        for alt_model in alternative_models:
            try:
                print(f"Attempting to load: {alt_model}")
                tokenizer = BertTokenizer.from_pretrained(alt_model)
                model = BertForSequenceClassification.from_pretrained(alt_model, num_labels=num_labels)
                print(f"Successfully loaded {alt_model}!")
                break
            except:
                continue
    
    return tokenizer, model

def load_model_from_local_path(model_path, num_labels=3):
    """Loads model from a specific local path (alternative method)."""
    print(f"Loading model from local path: {model_path}")
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    
    print("Model loaded successfully from local path!")
    return tokenizer, model

def save_model(model, tokenizer, save_path='./fine_tuned_model'):
    """Saves the fine-tuned model and tokenizer."""
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to: {save_path}")

def load_finetuned_model(model_path='./fine_tuned_model'):
    """Loads a fine-tuned model and tokenizer from a local path."""
    print(f"Loading fine-tuned model from: {model_path}")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    print("Fine-tuned model loaded successfully!")
    return tokenizer, model

def predict_sentiment(text, model, tokenizer, device='cpu'):
    """Predicts sentiment for a single text string."""
    model.eval()
    model = model.to(device)
    
    # Clean the text
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and encode
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    _, prediction = torch.max(logits, dim=1)
    
    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = label_map[prediction.item()]
    confidence = probabilities[0][prediction.item()].item()
    
    return sentiment, confidence

# Example usage
if __name__ == "__main__":
    # Method 1: Load from Hugging Face cache (RECOMMENDED)
    try:
        tokenizer, model = load_model('yiyanghkust/finbert-pretrain')
    except:
        # Method 2: Load from explicit local path (if Method 1 fails)
        model_path = r"C:\Users\samsh\.cache\huggingface\hub\models--yiyanghkust--finbert-pretrain\snapshots\88ab954a39ea6d3ce2b62cff086dd5ad1172c664"
        tokenizer, model = load_model_from_local_path(model_path)
    
    # Test prediction
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_text = "The company reported strong earnings growth and exceeded market expectations."
    
    sentiment, confidence = predict_sentiment(test_text, model, tokenizer, device)
    print(f"\nTest Text: {test_text}")
    print(f"Predicted Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2%}")