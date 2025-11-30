"""
Simple Daily Stock Scanner
Run this every morning to get today's stock recommendations!

Usage: python daily_stock_scanner.py
"""

import pandas as pd
import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
from sentiment_analyzer import load_finetuned_model, predict_sentiment
import torch
from tqdm import tqdm

# Popular stock lists
TECH_STOCKS = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Google', 
    'AMZN': 'Amazon', 'META': 'Meta', 'NVDA': 'Nvidia',
    'TSLA': 'Tesla', 'AMD': 'AMD', 'INTC': 'Intel', 'NFLX': 'Netflix'
}

FINANCIAL_STOCKS = {
    'JPM': 'JPMorgan', 'BAC': 'Bank of America', 'WFC': 'Wells Fargo',
    'GS': 'Goldman Sachs', 'MS': 'Morgan Stanley', 'C': 'Citigroup'
}

POPULAR_STOCKS = {**TECH_STOCKS, **FINANCIAL_STOCKS}


def get_stock_news_simple(ticker):
    """Fetch news from Yahoo Finance RSS (most reliable)."""
    try:
        url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
        feed = feedparser.parse(url)
        
        headlines = []
        for entry in feed.entries[:5]:
            headlines.append({
                'ticker': ticker,
                'headline': entry.title,
                'date': entry.published if hasattr(entry, 'published') else 'Today'
            })
        
        return headlines
    except:
        return []


def analyze_stocks_today(stock_dict):
    """Analyze stocks based on today's news."""
    
    # Load model
    print("Loading sentiment model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_finetuned_model('../models/finbert_sentiment_model')
    model.to(device)
    print("✓ Model loaded!\n")
    
    results = []
    
    print(f"Fetching and analyzing news for {len(stock_dict)} stocks...\n")
    
    for ticker, company in tqdm(stock_dict.items()):
        # Get news
        news_items = get_stock_news_simple(ticker)
        
        if not news_items:
            continue
        
        # Analyze each headline
        sentiments = []
        for news in news_items:
            sentiment, confidence = predict_sentiment(
                news['headline'], model, tokenizer, device
            )
            
            score = {'Positive': 1, 'Neutral': 0, 'Negative': -1}[sentiment]
            sentiments.append({
                'score': score,
                'confidence': confidence,
                'weighted': score * confidence
            })
        
        # Calculate average
        avg_weighted = sum(s['weighted'] for s in sentiments) / len(sentiments)
        avg_confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)
        
        # Determine action
        if avg_weighted > 0.3 and avg_confidence > 0.7:
            action = 'BUY'
        elif avg_weighted < -0.3 and avg_confidence > 0.7:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        results.append({
            'ticker': ticker,
            'company': company,
            'avg_score': avg_weighted,
            'confidence': avg_confidence,
            'action': action,
            'news_count': len(news_items),
            'top_headline': news_items[0]['headline'] if news_items else 'N/A'
        })
        
        time.sleep(0.5)  # Be nice to servers
    
    return pd.DataFrame(results)


def print_daily_report(df):
    """Print today's stock recommendations with detailed tables."""
    
    print("\n" + "="*100)
    print(f"{'DAILY STOCK REPORT':^100}")
    print(f"{datetime.now().strftime('%A, %B %d, %Y - %I:%M %p'):^100}")
    print("="*100)
    
    # ==================== SUMMARY TABLE ====================
    print("\n" + "="*100)
    print(f"{'COMPLETE STOCK ANALYSIS TABLE':^100}")
    print("="*100)
    print(f"{'Ticker':<8} {'Company':<20} {'Action':<8} {'Score':<10} {'Confidence':<12} {'News':<6}")
    print("-"*100)
    
    # Sort by action (BUY first, then HOLD, then SELL)
    action_order = {'BUY': 1, 'HOLD': 2, 'SELL': 3}
    df_sorted = df.sort_values(by=['action', 'avg_score'], 
                                key=lambda x: x.map(action_order) if x.name == 'action' else x,
                                ascending=[True, False])
    
    for idx, row in df_sorted.iterrows():
        # Color coding with emoji
        if row['action'] == 'BUY':
            emoji = '📈'
            action_display = f"{emoji} {row['action']}"
        elif row['action'] == 'SELL':
            emoji = '📉'
            action_display = f"{emoji} {row['action']}"
        else:
            emoji = '⏸️'
            action_display = f"{emoji} {row['action']}"
        
        print(f"{row['ticker']:<8} {row['company']:<20} {action_display:<9} "
              f"{row['avg_score']:+.3f}      {row['confidence']:.1%}        {row['news_count']}")
    
    print("="*100)
    
    # ==================== BUY RECOMMENDATIONS ====================
    buys = df[df['action'] == 'BUY'].sort_values('avg_score', ascending=False)
    
    if len(buys) > 0:
        print(f"\n{'='*100}")
        print(f"📈 STOCKS TO BUY TODAY ({len(buys)} opportunities)")
        print("="*100)
        print(f"{'Rank':<6} {'Ticker':<8} {'Company':<20} {'Score':<10} {'Confidence':<12} {'Top Headline'}")
        print("-"*100)
        
        for rank, (idx, row) in enumerate(buys.iterrows(), 1):
            headline = row['top_headline'][:60] + '...' if len(row['top_headline']) > 60 else row['top_headline']
            print(f"{rank:<6} {row['ticker']:<8} {row['company']:<20} {row['avg_score']:+.3f}      "
                  f"{row['confidence']:.1%}        {headline}")
        print("="*100)
    else:
        print(f"\n{'='*100}")
        print(f"📈 STOCKS TO BUY TODAY")
        print("="*100)
        print("No BUY opportunities found today. Market sentiment is cautious.")
        print("="*100)
    
    # ==================== HOLD RECOMMENDATIONS ====================
    holds = df[df['action'] == 'HOLD'].sort_values('avg_score', ascending=False)
    
    if len(holds) > 0:
        print(f"\n{'='*100}")
        print(f"⏸️  STOCKS TO HOLD/MONITOR ({len(holds)} stocks)")
        print("="*100)
        print(f"{'Ticker':<8} {'Company':<20} {'Score':<10} {'Confidence':<12} {'Status'}")
        print("-"*100)
        
        for idx, row in holds.iterrows():
            if row['avg_score'] > 0:
                status = "Slightly Positive"
            elif row['avg_score'] < 0:
                status = "Slightly Negative"
            else:
                status = "Neutral"
            
            print(f"{row['ticker']:<8} {row['company']:<20} {row['avg_score']:+.3f}      "
                  f"{row['confidence']:.1%}        {status}")
        print("="*100)
    
    # ==================== SELL RECOMMENDATIONS ====================
    sells = df[df['action'] == 'SELL'].sort_values('avg_score')
    
    if len(sells) > 0:
        print(f"\n{'='*100}")
        print(f"📉 STOCKS TO AVOID/SELL TODAY ({len(sells)} warnings)")
        print("="*100)
        print(f"{'Rank':<6} {'Ticker':<8} {'Company':<20} {'Score':<10} {'Confidence':<12} {'Top Headline'}")
        print("-"*100)
        
        for rank, (idx, row) in enumerate(sells.iterrows(), 1):
            headline = row['top_headline'][:60] + '...' if len(row['top_headline']) > 60 else row['top_headline']
            print(f"{rank:<6} {row['ticker']:<8} {row['company']:<20} {row['avg_score']:+.3f}      "
                  f"{row['confidence']:.1%}        {headline}")
        print("="*100)
    else:
        print(f"\n{'='*100}")
        print(f"📉 STOCKS TO AVOID TODAY")
        print("="*100)
        print("No strong SELL signals detected. No immediate risks identified.")
        print("="*100)
    
    # ==================== SUMMARY STATISTICS ====================
    total = len(df)
    print(f"\n{'='*100}")
    print(f"{'SUMMARY STATISTICS':^100}")
    print("="*100)
    print(f"Total Stocks Analyzed: {total}")
    print(f"  📈 BUY:  {len(buys):<3} ({len(buys)/total*100:>5.1f}%)  - Strong positive sentiment, recommended to buy")
    print(f"  ⏸️  HOLD: {len(holds):<3} ({len(holds)/total*100:>5.1f}%)  - Neutral or uncertain, monitor closely")
    print(f"  📉 SELL: {len(sells):<3} ({len(sells)/total*100:>5.1f}%)  - Strong negative sentiment, avoid or sell")
    print("="*100)
    
    # Average sentiment
    avg_market_score = df['avg_score'].mean()
    avg_market_conf = df['confidence'].mean()
    
    print(f"\nOverall Market Sentiment:")
    print(f"  Average Score: {avg_market_score:+.3f}")
    print(f"  Average Confidence: {avg_market_conf:.1%}")
    
    if avg_market_score > 0.2:
        market_mood = "🟢 BULLISH - Positive news dominates"
    elif avg_market_score < -0.2:
        market_mood = "🔴 BEARISH - Negative news dominates"
    else:
        market_mood = "🟡 NEUTRAL - Mixed signals, cautious approach"
    
    print(f"  Market Mood: {market_mood}")
    print("="*100)


def main():
    print("\n" + "="*100)
    print(f"{'REAL-TIME STOCK NEWS ANALYZER':^100}")
    print("="*100)
    
    # Let user choose stock list
    print("\nSelect stock list to analyze:")
    print("1. Tech Stocks (10 stocks)")
    print("2. Financial Stocks (6 stocks)")
    print("3. All Popular Stocks (16 stocks)")
    print("4. Custom (enter tickers manually)")
    
    choice = input("\nEnter choice (1-4) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        stocks = TECH_STOCKS
    elif choice == "2":
        stocks = FINANCIAL_STOCKS
    elif choice == "3":
        stocks = POPULAR_STOCKS
    elif choice == "4":
        tickers = input("Enter tickers separated by commas (e.g., AAPL,MSFT,GOOGL): ").strip()
        stocks = {t.strip(): t.strip() for t in tickers.split(',')}
    else:
        stocks = TECH_STOCKS
    
    # Analyze
    results_df = analyze_stocks_today(stocks)
    
    # Print report
    print_daily_report(results_df)
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f"../results/daily_stocks_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    print(f"\n✓ Full results saved to: {filename}")
    
    # Show top 3 BUY opportunities
    top_buys = results_df[results_df['action'] == 'BUY'].nlargest(3, 'avg_score')
    
    if len(top_buys) > 0:
        print("\n" + "="*100)
        print(f"{'🏆 TOP 3 INVESTMENT OPPORTUNITIES TODAY':^100}")
        print("="*100)
        for i, (_, row) in enumerate(top_buys.iterrows(), 1):
            print(f"{i}. {row['ticker']} ({row['company']})")
            print(f"   Score: {row['avg_score']:+.3f} | Confidence: {row['confidence']:.1%}")
            print(f"   Latest: {row['top_headline'][:80]}...")
            print()
        print("="*100)
    
    print("\n💡 TIP: Run this script every morning before market opens for best results!")
    print("="*100)


if __name__ == '__main__':
    main()