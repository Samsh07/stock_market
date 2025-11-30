import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import requests
from bs4 import BeautifulSoup
import os

def load_data(file_path):
    """Loads data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return pd.read_csv(file_path)

def clean_text(text):
    """Cleans and preprocesses text data."""
    text = str(text)  # Convert to string in case of non-string input
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(df):
    """Applies cleaning and drops missing values."""
    df = df.dropna(subset=['Content', 'Sentiment'])
    df['Content'] = df['Content'].apply(clean_text)
    
    # Remove empty strings after cleaning
    df = df[df['Content'].str.len() > 0]
    
    return df

def split_data(df, test_size=0.2, val_size=0.25, random_state=42):
    """Splits data into train, validation, and test sets."""
    # Map sentiment strings to integers (case-insensitive)
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['Sentiment'] = df['Sentiment'].str.lower().map(label_map)
    
    # Remove any unmapped sentiments
    df = df.dropna(subset=['Sentiment'])
    df['Sentiment'] = df['Sentiment'].astype(int)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['Sentiment']
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=random_state, stratify=train_df['Sentiment']
    )
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    return train_df, val_df, test_df

def scrape_financial_news(url):
    """Scrapes title and content from a financial news URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title = soup.find('h1').get_text() if soup.find('h1') else ''
        
        # Common content selectors
        content_selectors = [
            'div.article-body',
            'div.article-content',
            'div.story-content',
            'div.content',
            'article'
        ]
        
        content = ''
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                content = content_element.get_text()
                break
        
        if not content:
            # Fallback to all paragraphs
            paragraphs = soup.find_all('p')
            content = '\n'.join([p.get_text() for p in paragraphs])
        
        return title, content
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return '', ''

def save_data(df, file_path):
    """Saves a DataFrame to a CSV file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Data saved to: {file_path}")