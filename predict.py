import torch

# Import from correct modules
from sentiment_analyzer import load_finetuned_model, predict_sentiment

def main():
    MODEL_PATH = '../models/finbert_sentiment_model'  # FIXED: Added ../
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*50)
    print("Financial Sentiment Analysis - Prediction")
    print("="*50)
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading model...")
    tokenizer, model = load_finetuned_model(MODEL_PATH)
    model.to(device)
    print("Model loaded successfully!\n")
    
    # Test cases
    test_cases = [
        "The company reported a record profit, exceeding all expectations.",
        "Shares plummeted after the company announced disappointing quarterly results.",
        "The stock price remained stable throughout the trading session.",
        "Investors are optimistic about the upcoming product launch.",
        "The company faces bankruptcy amid mounting debt and declining revenue."
    ]
    
    print("Analyzing sample texts...\n")
    print("="*50)
    
    for i, sample_text in enumerate(test_cases, 1):
        sentiment, confidence = predict_sentiment(sample_text, model, tokenizer, device)
        print(f"\nExample {i}:")
        print(f"Text: {sample_text}")
        print(f"Predicted Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.2%}")
        print("-"*50)
    
    # Interactive mode
    print("\n" + "="*50)
    print("Interactive Mode")
    print("="*50)
    print("Enter financial news text to analyze (or 'quit' to exit):\n")
    
    while True:
        user_input = input("\nEnter text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting prediction mode. Goodbye!")
            break
        
        if not user_input:
            print("Please enter some text.")
            continue
        
        sentiment, confidence = predict_sentiment(user_input, model, tokenizer, device)
        print(f"\nPredicted Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.2%}")

if __name__ == '__main__':
    main()