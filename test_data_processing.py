import unittest
import pandas as pd
import sys
import os

# FIXED: Remove sys.path modification if all files are in the same directory
# If you have a 'tests' subdirectory, keep this but adjust the import

# FIXED: Import from correct modules (no 'src' prefix)
from data_processing import clean_text, preprocess_data, split_data

class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        # Create a simple test dataframe
        self.test_df = pd.DataFrame({
            'Content': [
                'This is a positive news article about profits.',
                'This is a negative news about losses.',
                'This is a neutral news report.',
                'Another positive article with http://example.com link!',
                'Another negative article @#$% with special chars.',
                'Another neutral report.'
            ],  # FIXED: Added more samples for better split testing
            'Sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative', 'neutral']  # FIXED: lowercase to match preprocess_data
        })
    
    def test_clean_text(self):
        """Test text cleaning."""
        text = "This is a test with http://example.com and special characters!"
        cleaned = clean_text(text)
        
        # Check that URL is removed
        self.assertNotIn('http://example.com', cleaned)
        
        # Check that special characters are removed
        self.assertNotIn('!', cleaned)
        
        # Check that regular words remain
        self.assertIn('test', cleaned)
        self.assertIn('with', cleaned)
        
        # FIXED: 'special' should still be in the text after cleaning
        self.assertIn('special', cleaned)
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        processed = preprocess_data(self.test_df)
        
        # Check that all rows are preserved (no missing values)
        self.assertEqual(len(processed), 6)
        
        # Check that required columns exist
        self.assertIn('Content', processed.columns)
        self.assertIn('Sentiment', processed.columns)
        
        # Check that content is cleaned (no special characters)
        for content in processed['Content']:
            self.assertNotIn('http://', content)
            self.assertNotIn('@', content)
            self.assertNotIn('#', content)
    
    def test_split_data(self):
        """Test data splitting."""
        processed = preprocess_data(self.test_df)
        train_df, val_df, test_df = split_data(processed, test_size=0.33, val_size=0.5, random_state=42)
        
        # Check that all data is accounted for
        total_samples = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total_samples, len(processed))
        
        # Check that all splits have at least one sample
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(val_df), 0)
        self.assertGreater(len(test_df), 0)
        
        # FIXED: Check that sentiment is converted to integers
        self.assertTrue(all(isinstance(x, (int, np.int64)) for x in train_df['Sentiment']))
        self.assertTrue(all(isinstance(x, (int, np.int64)) for x in val_df['Sentiment']))
        self.assertTrue(all(isinstance(x, (int, np.int64)) for x in test_df['Sentiment']))
        
        # Check that sentiment values are in expected range (0, 1, 2)
        all_sentiments = list(train_df['Sentiment']) + list(val_df['Sentiment']) + list(test_df['Sentiment'])
        self.assertTrue(all(s in [0, 1, 2] for s in all_sentiments))
    
    def test_clean_text_edge_cases(self):
        """Test text cleaning with edge cases."""
        # Empty string
        self.assertEqual(clean_text(''), '')
        
        # Only URLs
        self.assertEqual(clean_text('http://example.com'), '')
        
        # Only special characters
        cleaned = clean_text('!@#$%^&*()')
        self.assertEqual(cleaned.strip(), '')
        
        # Multiple spaces
        cleaned = clean_text('word1    word2     word3')
        self.assertEqual(cleaned, 'word1 word2 word3')
    
    def test_preprocess_data_with_missing_values(self):
        """Test preprocessing with missing values."""
        df_with_missing = pd.DataFrame({
            'Content': ['Valid content', None, 'Another valid content'],
            'Sentiment': ['positive', 'negative', None]
        })
        
        processed = preprocess_data(df_with_missing)
        
        # Should only have the valid row
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed.iloc[0]['Content'], 'Valid content')

if __name__ == '__main__':
    # FIXED: Add numpy import for testing
    import numpy as np
    
    # Run tests with verbose output
    unittest.main(verbosity=2)