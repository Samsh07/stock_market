"""
Indian Stock News Analyzer with Live Prices (NSE Stocks)
Fetches real-time prices and news for Indian stocks with BUY/HOLD/SELL recommendations

Usage: python indian_stock_scanner.py
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
import re
import json

# Top 50 NSE Stocks
INDIAN_STOCKS = {
    'RELIANCE': 'Reliance Industries Ltd',
    'HDFCBANK': 'HDFC Bank Ltd',
    'ICICIBANK': 'ICICI Bank Ltd',
    'SBIN': 'State Bank of India',
    'INFY': 'Infosys Ltd',
    'TCS': 'Tata Consultancy Services',
    'BHARTIARTL': 'Bharti Airtel Ltd',
    'LICI': 'Life Insurance Corporation',
    'HINDUNILVR': 'Hindustan Unilever Ltd',
    'BAJFINANCE': 'Bajaj Finance Ltd',
    'LT': 'Larsen & Toubro Ltd',
    'ITC': 'ITC Ltd',
    'MARUTI': 'Maruti Suzuki India Ltd',
    'KOTAKBANK': 'Kotak Mahindra Bank Ltd',
    'M&M': 'Mahindra & Mahindra Ltd',
    'HCLTECH': 'HCL Technologies Ltd',
    'SUNPHARMA': 'Sun Pharmaceutical',
    'AXISBANK': 'Axis Bank Ltd',
    'TITAN': 'Titan Company Ltd',
    'ULTRACEMCO': 'UltraTech Cement Ltd',
    'BAJAJFINSV': 'Bajaj Finserv Ltd',
    'ADANIPORTS': 'Adani Ports',
    'NTPC': 'NTPC Ltd',
    'ADANIENT': 'Adani Enterprises Ltd',
    'ONGC': 'ONGC Ltd',
    'APOLLOHOSP': 'Apollo Hospitals',
    'GRASIM': 'Grasim Industries Ltd',
    'WIPRO': 'Wipro Ltd',
    'ASIANPAINT': 'Asian Paints Ltd',
    'DRREDDY': 'Dr Reddys Laboratories',
    'PIDILITIND': 'Pidilite Industries Ltd',
    'INDUSINDBK': 'IndusInd Bank Ltd',
    'POWERGRID': 'Power Grid Corporation',
    'COALINDIA': 'Coal India Ltd',
    'JSWSTEEL': 'JSW Steel Ltd',
    'TECHM': 'Tech Mahindra Ltd',
    'TATASTEEL': 'Tata Steel Ltd',
    'IOC': 'Indian Oil Corporation',
    'BPCL': 'Bharat Petroleum',
    'GAIL': 'GAIL India Ltd',
    'HINDALCO': 'Hindalco Industries Ltd',
    'CIPLA': 'Cipla Ltd',
    'UPL': 'UPL Ltd',
    'DIVISLAB': 'Divis Laboratories Ltd',
    'HINDZINC': 'Hindustan Zinc Ltd',
    'TATAMOTORS': 'Tata Motors Ltd',
    'EICHERMOT': 'Eicher Motors Ltd',
    'ADANIGREEN': 'Adani Green Energy Ltd',
    'HDFCLIFE': 'HDFC Life Insurance',
    'BRITANNIA': 'Britannia Industries Ltd'
}


def get_stock_price(ticker):
    """
    Fetch real-time stock price from Yahoo Finance.
    Returns: dict with price, change, change_percent, market_status
    """
    try:
        # Yahoo Finance uses .NS suffix for NSE stocks
        yahoo_ticker = f"{ticker}.NS"
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_ticker}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        
        if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
            result = data['chart']['result'][0]
            meta = result['meta']
            
            current_price = meta.get('regularMarketPrice', 0)
            previous_close = meta.get('previousClose', 0)
            
            if current_price and previous_close:
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100
                
                # Check market status
                market_state = meta.get('marketState', 'CLOSED')
                
                return {
                    'price': round(current_price, 2),
                    'change': round(change, 2),
                    'change_percent': round(change_percent, 2),
                    'previous_close': round(previous_close, 2),
                    'market_status': market_state,
                    'currency': meta.get('currency', 'INR'),
                    'success': True
                }
        
        return {'success': False, 'price': 0, 'change': 0, 'change_percent': 0}
        
    except Exception as e:
        return {'success': False, 'price': 0, 'change': 0, 'change_percent': 0, 'error': str(e)}


def get_indian_stock_news(ticker, company_name):
    """Fetch news for Indian stocks from multiple sources."""
    news_items = []
    
    # Method 1: Google News (most reliable for Indian stocks)
    try:
        query = f"{company_name} stock India NSE".replace(' ', '+')
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]:
            news_items.append({
                'ticker': ticker,
                'headline': entry.title,
                'link': entry.link,
                'published': entry.published if hasattr(entry, 'published') else 'Today',
                'source': 'Google News'
            })
    except Exception as e:
        print(f"  ⚠ Error fetching Google News for {ticker}: {e}")
    
    # Method 2: Economic Times RSS
    try:
        query = company_name.replace(' ', '+')
        url = f"https://economictimes.indiatimes.com/rssfeedsdefault.cms"
        
        feed = feedparser.parse(url)
        for entry in feed.entries[:20]:
            if any(word in entry.title.lower() for word in ticker.lower().split()):
                news_items.append({
                    'ticker': ticker,
                    'headline': entry.title,
                    'link': entry.link,
                    'published': entry.published if hasattr(entry, 'published') else 'Today',
                    'source': 'Economic Times'
                })
                if len([n for n in news_items if n['source'] == 'Economic Times']) >= 3:
                    break
    except:
        pass
    
    # Remove duplicates
    unique_news = []
    seen = set()
    for news in news_items:
        headline_clean = news['headline'].lower().strip()
        if headline_clean not in seen:
            seen.add(headline_clean)
            unique_news.append(news)
    
    return unique_news[:8]


def analyze_indian_stocks(stock_dict, max_stocks=None):
    """Analyze Indian stocks with prices and news sentiment."""
    
    if max_stocks:
        stock_dict = dict(list(stock_dict.items())[:max_stocks])
    
    # Load model
    print("Loading sentiment model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_finetuned_model('../models/finbert_sentiment_model')
    model.to(device)
    print("✓ Model loaded!\n")
    
    results = []
    
    print(f"Fetching prices and analyzing news for {len(stock_dict)} Indian stocks...\n")
    
    for ticker, company in tqdm(stock_dict.items(), desc="Processing"):
        # Get stock price
        price_data = get_stock_price(ticker)
        
        # Get news
        news_items = get_indian_stock_news(ticker, company)
        
        if not news_items or len(news_items) < 2:
            # Still save stock with price data even if no news
            if price_data['success']:
                results.append({
                    'ticker': ticker,
                    'company': company,
                    'price': price_data['price'],
                    'change': price_data['change'],
                    'change_percent': price_data['change_percent'],
                    'market_status': price_data['market_status'],
                    'avg_score': 0,
                    'confidence': 0,
                    'action': 'HOLD',
                    'positive_count': 0,
                    'neutral_count': 0,
                    'negative_count': 0,
                    'news_count': 0,
                    'top_headline': 'No recent news available'
                })
            continue
        
        # Analyze sentiment
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
        
        positive_count = sum(1 for s in sentiments if s['score'] > 0)
        negative_count = sum(1 for s in sentiments if s['score'] < 0)
        neutral_count = sum(1 for s in sentiments if s['score'] == 0)
        
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
            'price': price_data['price'] if price_data['success'] else 'N/A',
            'change': price_data['change'] if price_data['success'] else 0,
            'change_percent': price_data['change_percent'] if price_data['success'] else 0,
            'market_status': price_data.get('market_status', 'UNKNOWN'),
            'avg_score': avg_weighted,
            'confidence': avg_confidence,
            'action': action,
            'positive_count': positive_count,
            'neutral_count': neutral_count,
            'negative_count': negative_count,
            'news_count': len(news_items),
            'top_headline': news_items[0]['headline'] if news_items else 'N/A'
        })
        
        time.sleep(0.5)
    
    return pd.DataFrame(results)


def print_indian_stock_report(df):
    """Print detailed report with prices for Indian stocks."""
    
    print("\n" + "="*130)
    print(f"{'INDIAN STOCK MARKET ANALYSIS WITH LIVE PRICES (NSE)':^130}")
    print(f"{datetime.now().strftime('%A, %B %d, %Y - %I:%M %p IST'):^130}")
    print("="*130)
    
    if df.empty:
        print("\n⚠ No stocks analyzed. Please check your internet connection.")
        return
    
    # Market status indicator
    market_statuses = df['market_status'].value_counts()
    if 'REGULAR' in market_statuses.index or 'PRE' in market_statuses.index or 'POST' in market_statuses.index:
        print(f"\n🟢 Market Status: OPEN")
    else:
        print(f"\n🔴 Market Status: CLOSED (Showing last traded prices)")
    
    # ==================== COMPLETE TABLE ====================
    print("\n" + "="*130)
    print(f"{'COMPLETE ANALYSIS TABLE WITH PRICES':^130}")
    print("="*130)
    print(f"{'Ticker':<12} {'Company':<24} {'Price (₹)':<12} {'Change':<10} {'Action':<8} {'Score':<8} {'+':<3} {'=':<3} {'-':<3}")
    print("-"*130)
    
    action_order = {'BUY': 1, 'HOLD': 2, 'SELL': 3}
    df_sorted = df.sort_values(
        by=['action', 'avg_score'],
        key=lambda x: x.map(action_order) if x.name == 'action' else x,
        ascending=[True, False]
    )
    
    for _, row in df_sorted.iterrows():
        emoji = {'BUY': '📈', 'HOLD': '⏸️', 'SELL': '📉'}[row['action']]
        action_display = f"{emoji} {row['action']}"
        
        company_short = row['company'][:22] + '..' if len(row['company']) > 24 else row['company']
        
        # Format price and change
        if row['price'] != 'N/A':
            price_str = f"₹{row['price']:,.2f}"
            
            if row['change_percent'] > 0:
                change_str = f"🟢 +{row['change_percent']:.2f}%"
            elif row['change_percent'] < 0:
                change_str = f"🔴 {row['change_percent']:.2f}%"
            else:
                change_str = f"⚪ 0.00%"
        else:
            price_str = "N/A"
            change_str = "N/A"
        
        print(f"{row['ticker']:<12} {company_short:<24} {price_str:<12} {change_str:<10} {action_display:<9} "
              f"{row['avg_score']:+.3f}    {int(row['positive_count']):<3} {int(row['neutral_count']):<3} {int(row['negative_count']):<3}")
    
    print("="*130)
    
    # ==================== BUY RECOMMENDATIONS ====================
    buys = df[df['action'] == 'BUY'].sort_values('avg_score', ascending=False)
    
    print(f"\n{'='*130}")
    print(f"📈 STOCKS TO BUY TODAY ({len(buys)} opportunities)")
    print("="*130)
    
    if len(buys) > 0:
        print(f"{'Rank':<6} {'Ticker':<12} {'Company':<26} {'Price (₹)':<14} {'Change':<12} {'Score':<10} {'Conf':<8}")
        print("-"*130)
        
        for rank, (_, row) in enumerate(buys.iterrows(), 1):
            company_short = row['company'][:24] + '..' if len(row['company']) > 26 else row['company']
            
            if row['price'] != 'N/A':
                price_str = f"₹{row['price']:,.2f}"
                change_str = f"+{row['change_percent']:.2f}%" if row['change_percent'] >= 0 else f"{row['change_percent']:.2f}%"
            else:
                price_str = "N/A"
                change_str = "N/A"
            
            print(f"{rank:<6} {row['ticker']:<12} {company_short:<26} {price_str:<14} {change_str:<12} "
                  f"{row['avg_score']:+.3f}      {row['confidence']:.1%}")
        
        print("\nTop Buy Recommendation Headlines:")
        for i, (_, row) in enumerate(buys.head(3).iterrows(), 1):
            print(f"{i}. {row['ticker']}: {row['top_headline'][:90]}...")
    else:
        print("No strong BUY signals detected today.")
    
    print("="*130)
    
    # ==================== HOLD TABLE ====================
    holds = df[df['action'] == 'HOLD'].sort_values('avg_score', ascending=False)
    
    if len(holds) > 0:
        print(f"\n{'='*130}")
        print(f"⏸️  STOCKS TO HOLD/MONITOR ({len(holds)} stocks)")
        print("="*130)
        print(f"{'Ticker':<12} {'Company':<26} {'Price (₹)':<14} {'Change':<12} {'Score':<10}")
        print("-"*130)
        
        for _, row in holds.iterrows():
            company_short = row['company'][:24] + '..' if len(row['company']) > 26 else row['company']
            
            if row['price'] != 'N/A':
                price_str = f"₹{row['price']:,.2f}"
                change_str = f"+{row['change_percent']:.2f}%" if row['change_percent'] >= 0 else f"{row['change_percent']:.2f}%"
            else:
                price_str = "N/A"
                change_str = "N/A"
            
            print(f"{row['ticker']:<12} {company_short:<26} {price_str:<14} {change_str:<12} {row['avg_score']:+.3f}")
        
        print("="*130)
    
    # ==================== SELL TABLE ====================
    sells = df[df['action'] == 'SELL'].sort_values('avg_score')
    
    if len(sells) > 0:
        print(f"\n{'='*130}")
        print(f"📉 STOCKS TO AVOID/SELL TODAY ({len(sells)} warnings)")
        print("="*130)
        print(f"{'Rank':<6} {'Ticker':<12} {'Company':<26} {'Price (₹)':<14} {'Change':<12} {'Score':<10} {'Conf':<8}")
        print("-"*130)
        
        for rank, (_, row) in enumerate(sells.iterrows(), 1):
            company_short = row['company'][:24] + '..' if len(row['company']) > 26 else row['company']
            
            if row['price'] != 'N/A':
                price_str = f"₹{row['price']:,.2f}"
                change_str = f"+{row['change_percent']:.2f}%" if row['change_percent'] >= 0 else f"{row['change_percent']:.2f}%"
            else:
                price_str = "N/A"
                change_str = "N/A"
            
            print(f"{rank:<6} {row['ticker']:<12} {company_short:<26} {price_str:<14} {change_str:<12} "
                  f"{row['avg_score']:+.3f}      {row['confidence']:.1%}")
        
        print("="*130)
    
    # ==================== SUMMARY ====================
    total = len(df)
    print(f"\n{'='*130}")
    print(f"{'SUMMARY STATISTICS':^130}")
    print("="*130)
    print(f"Total Stocks Analyzed: {total}")
    print(f"  📈 BUY:  {len(buys):<3} ({len(buys)/total*100:>5.1f}%)")
    print(f"  ⏸️  HOLD: {len(holds):<3} ({len(holds)/total*100:>5.1f}%)")
    print(f"  📉 SELL: {len(sells):<3} ({len(sells)/total*100:>5.1f}%)")
    
    # Price movement summary
    gainers = df[df['change_percent'] > 0]
    losers = df[df['change_percent'] < 0]
    
    print(f"\nPrice Movement:")
    print(f"  🟢 Gainers: {len(gainers)} stocks")
    print(f"  🔴 Losers: {len(losers)} stocks")
    
    if len(gainers) > 0:
        top_gainer = gainers.nlargest(1, 'change_percent').iloc[0]
        print(f"  📊 Top Gainer: {top_gainer['ticker']} (+{top_gainer['change_percent']:.2f}%)")
    
    if len(losers) > 0:
        top_loser = losers.nsmallest(1, 'change_percent').iloc[0]
        print(f"  📊 Top Loser: {top_loser['ticker']} ({top_loser['change_percent']:.2f}%)")
    
    avg_score = df['avg_score'].mean()
    print(f"\nOverall Market Sentiment Score: {avg_score:+.3f}")
    
    if avg_score > 0.2:
        print("Market Mood: 🟢 BULLISH")
    elif avg_score < -0.2:
        print("Market Mood: 🔴 BEARISH")
    else:
        print("Market Mood: 🟡 NEUTRAL")
    
    print("="*130)


def main():
    print("\n" + "="*130)
    print(f"{'INDIAN STOCK NEWS ANALYZER WITH LIVE PRICES (NSE)':^130}")
    print("="*130)
    
    print("\nSelect stocks to analyze:")
    print("1. Top 10 Stocks (Quick scan - ~30 seconds)")
    print("2. Top 25 Stocks (Medium scan - ~2 minutes)")
    print("3. All 50 Stocks (Full scan - ~5 minutes)")
    print("4. Custom selection (enter stock tickers)")
    
    choice = input("\nEnter choice (1-4) [default: 1]: ").strip() or "1"
    
    # Ask about email
    send_email = input("\n📧 Send report via email? (y/n) [default: n]: ").strip().lower()
    recipient_emails = []
    
    if send_email == 'y':
        emails_input = input("Enter recipient email(s) (comma-separated): ").strip()
        recipient_emails = [email.strip() for email in emails_input.split(',') if email.strip()]
    
    if choice == "1":
        stocks = dict(list(INDIAN_STOCKS.items())[:10])
        print(f"\n✓ Analyzing Top 10 Indian stocks...")
    elif choice == "2":
        stocks = dict(list(INDIAN_STOCKS.items())[:25])
        print(f"\n✓ Analyzing Top 25 Indian stocks...")
    elif choice == "3":
        stocks = INDIAN_STOCKS
        print(f"\n✓ Analyzing All 50 Indian stocks...")
    elif choice == "4":
        tickers = input("Enter NSE tickers (e.g., RELIANCE,TCS,INFY): ").strip().upper()
        custom_stocks = {}
        for ticker in tickers.split(','):
            ticker = ticker.strip()
            if ticker in INDIAN_STOCKS:
                custom_stocks[ticker] = INDIAN_STOCKS[ticker]
            else:
                print(f"⚠ Warning: {ticker} not found")
        stocks = custom_stocks if custom_stocks else dict(list(INDIAN_STOCKS.items())[:10])
    else:
        stocks = dict(list(INDIAN_STOCKS.items())[:10])
    
    # Analyze
    results_df = analyze_indian_stocks(stocks)
    
    # Print report
    print_indian_stock_report(results_df)
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"../results/indian_stocks_with_prices_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    print(f"\n✓ Full results saved to: {filename}")
    
    # Top opportunities with prices
    top_buys = results_df[results_df['action'] == 'BUY'].nlargest(3, 'avg_score')
    
    if len(top_buys) > 0:
        print("\n" + "="*130)
        print(f"{'🏆 TOP 3 INVESTMENT OPPORTUNITIES TODAY':^130}")
        print("="*130)
        for i, (_, row) in enumerate(top_buys.iterrows(), 1):
            print(f"\n{i}. {row['ticker']} - {row['company']}")
            if row['price'] != 'N/A':
                print(f"   💰 Current Price: ₹{row['price']:,.2f} ({row['change_percent']:+.2f}%)")
            print(f"   📊 Sentiment Score: {row['avg_score']:+.3f} | Confidence: {row['confidence']:.1%}")
            print(f"   📰 News: {row['positive_count']}+ / {row['neutral_count']}= / {row['negative_count']}-")
        print("="*130)
    
    print("\n💡 TIP: Run during market hours (9:15 AM - 3:30 PM IST) for live prices!")
    print("="*130)


if __name__ == '__main__':
    main()