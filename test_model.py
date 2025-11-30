import unittest
import torch
import pandas as pd
from sentiment_analyzer import FinancialNewsDataset, predict_sentiment, load_model
from transformers import BertTokenizer

class TestModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load model once for all tests."""
        print("\nLoading model for testing...")
        try:
            cls.tokenizer, cls.model = load_model('yiyanghkust/finbert-pretrain', num_labels=3)
            cls.device = 'cpu'  # Use CPU for testing
            cls.model.to(cls.device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            cls.tokenizer = None
            cls.model = None
    
    def setUp(self):
        """Skip tests if model is not loaded."""
        if self.model is None or self.tokenizer is None:
            self.skipTest("Model not available for testing")
    
    def test_dataset_creation(self):
        """Test custom dataset creation."""
        df = pd.DataFrame({
            'Content': ['This is positive news', 'This is negative news'],
            'Sentiment': [2, 0]  # 0=Negative, 1=Neutral, 2=Positive
        })
        
        dataset = FinancialNewsDataset(df, self.tokenizer, max_token_length=128)
        
        # Check dataset length
        self.assertEqual(len(dataset), 2)
        
        # Check first item structure
        item = dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('labels', item)
        
        # Check tensor shapes
        self.assertEqual(item['input_ids'].shape[0], 128)
        self.assertEqual(item['attention_mask'].shape[0], 128)
    
    def test_predict_sentiment(self):
        """Test sentiment prediction."""
        test_cases = [
            ("The company reported strong earnings growth", "Positive"),
            ("The stock crashed after poor results", "Negative"),
            ("The market remained stable today", "Neutral")
        ]
        
        for text, expected_sentiment in test_cases:
            sentiment, confidence = predict_sentiment(text, self.model, self.tokenizer, self.device)
            
            # Check that sentiment is one of the expected values
            self.assertIn(sentiment, ['Negative', 'Neutral', 'Positive'])
            
            # Check that confidence is between 0 and 1
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
            
            print(f"\nText: {text}")
            print(f"Predicted: {sentiment} (Confidence: {confidence:.2%})")
    
    def test_predict_sentiment_edge_cases(self):
        """Test prediction with edge cases."""
        edge_cases = [
            "",  # Empty string
            "   ",  # Only spaces
            "a",  # Single character
            "The " * 200  # Very long text
        ]
        
        for text in edge_cases:
            try:
                sentiment, confidence = predict_sentiment(text, self.model, self.tokenizer, self.device)
                self.assertIn(sentiment, ['Negative', 'Neutral', 'Positive'])
            except Exception as e:
                self.fail(f"Prediction failed for edge case: {text[:50]}... Error: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)