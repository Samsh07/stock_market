# check_data.py
import pandas as pd

print("Checking processed data...\n")

# FIXED: Use correct relative paths from src folder
train_df = pd.read_csv('../data/processed/train.csv')
val_df = pd.read_csv('../data/processed/val.csv')
test_df = pd.read_csv('../data/processed/test.csv')

print(f"Train samples: {len(train_df)}")
print(f"Val samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Total samples: {len(train_df) + len(val_df) + len(test_df)}")

print("\nTrain columns:", train_df.columns.tolist())
print("\nSample from train data:")
print(train_df.head(2))

print("\nSentiment distribution in train:")
print(train_df['Sentiment'].value_counts())

print("\nSentiment distribution in val:")
print(val_df['Sentiment'].value_counts())

print("\nSentiment distribution in test:")
print(test_df['Sentiment'].value_counts())

print("\nChecking for missing values:")
print(f"Train: {train_df.isnull().sum().sum()}")
print(f"Val: {val_df.isnull().sum().sum()}")
print(f"Test: {test_df.isnull().sum().sum()}")

print("\nData types:")
print(train_df.dtypes)

print("\nChecking if sentiments are integers (0, 1, 2):")
unique_sentiments = sorted(train_df['Sentiment'].unique())
print(f"Unique sentiments in train: {unique_sentiments}")
print(f"Expected: [0, 1, 2] (0=Negative, 1=Neutral, 2=Positive)")