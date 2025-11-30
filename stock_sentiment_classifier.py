"""
Real-time Stock News Analyzer with Enhanced Table Display
Fetches today's financial news and analyzes sentiment for stock recommendations.

Requires: pip install feedparser beautifulsoup4 requests
"""

import pandas as pd
import requests
from datetime import datetime
import time
from bs4 import BeautifulSoup
import feedparser
from sentiment_analyzer import load_finetuned_model, predict_sentiment
import torch
from tqdm import tqdm

class RealTimeStockNewsAnalyzer:
    """Fetch and analyze real-time stock news."""
    
    def __init__(self, model_path='../models/finbert_sentiment_model'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading sentiment model on {self.device}...")
        self.tokenizer, self.model = load_finetuned_model(model_path)
        self.model.to(self.device)
        print("✓ Model loaded!\n")
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def get_yahoo_finance_news(self, ticker):
        """Fetch news from Yahoo Finance RSS feed."""
        try:
            url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
            feed = feedparser.parse(url)
            
            news_items = []
            for entry in feed.entries[:5]:
                news_items.append({
                    'ticker': ticker,
                    'headline': entry.title,
                    'link': entry.link,
                    'published': entry.published if hasattr(entry, 'published') else 'N/A',
                    'source': 'Yahoo Finance'
                })
            return news_items
        except Exception as e:
            return []
    
    def get_finviz_news(self, ticker):
        """Fetch news from Finviz."""
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            news_items = []
            news_table = soup.find(id='news-table')
            
            if news_table:
                for row in news_table.find_all('tr')[:5]:
                    title = row.a.text if row.a else ''
                    link = row.a['href'] if row.a else ''
                    date_data = row.td.text.strip() if row.td else ''
                    
                    news_items.append({
                        'ticker': ticker,
                        'headline': title,
                        'link': link,
                        'published': date_data,
                        'source': 'Finviz'
                    })
            
            return news_items
        except Exception as e:
            return []
    
    def get_google_news(self, ticker, company_name=None):
        """Fetch news from Google News RSS."""
        try:
            query = company_name if company_name else ticker
            url = f"https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            
            news_items = []
            for entry in feed.entries[:5]:
                news_items.append({
                    'ticker': ticker,
                    'headline': entry.title,
                    'link': entry.link,
                    'published': entry.published if hasattr(entry, 'published') else 'N/A',
                    'source': 'Google News'
                })
            return news_items
        except Exception as e:
            return []
    
    def fetch_all_news_for_stock(self, ticker, company_name=None, max_news=10):
        """Fetch news from all available sources."""
        print(f"Fetching news for {ticker}...")
        
        all_news = []
        all_news.extend(self.get_yahoo_finance_news(ticker))
        time.sleep(0.5)
        all_news.extend(self.get_finviz_news(ticker))
        time.sleep(0.5)
        all_news.extend(self.get_google_news(ticker, company_name))
        time.sleep(0.5)
        
        unique_news = []
        seen_headlines = set()
        
        for news in all_news:
            headline_lower = news['headline'].lower()
            if headline_lower not in seen_headlines:
                seen_headlines.add(headline_lower)
                unique_news.append(news)
        
        return unique_news[:max_news]
    
    def analyze_stock_news(self, ticker, company_name=None, max_news=10):
        """Fetch and analyze news for a single stock."""
        news_items = self.fetch_all_news_for_stock(ticker, company_name, max_news)
        
        if not news_items:
            print(f"  ⚠ No news found for {ticker}")
            return None
        
        print(f"  ✓ Found {len(news_items)} news articles")
        
        analyzed_news = []
        for news in news_items:
            sentiment, confidence = predict_sentiment(
                news['headline'], self.model, self.tokenizer, self.device
            )
            
            news['sentiment'] = sentiment
            news['confidence'] = confidence
            news['sentiment_score'] = {'Positive': 1, 'Neutral': 0, 'Negative': -1}[sentiment]
            analyzed_news.append(news)
        
        avg_score = sum(n['sentiment_score'] * n['confidence'] for n in analyzed_news) / len(analyzed_news)
        avg_confidence = sum(n['confidence'] for n in analyzed_news) / len(analyzed_news)
        
        positive_count = sum(1 for n in analyzed_news if n['sentiment'] == 'Positive')
        negative_count = sum(1 for n in analyzed_news if n['sentiment'] == 'Negative')
        neutral_count = sum(1 for n in analyzed_news if n['sentiment'] == 'Neutral')
        
        if avg_score > 0.3 and avg_confidence > 0.7:
            recommendation = 'BUY'
        elif avg_score < -0.3 and avg_confidence > 0.7:
            recommendation = 'SELL'
        else:
            recommendation = 'HOLD'
        
        return {
            'ticker': ticker,
            'company_name': company_name or ticker,
            'total_news': len(analyzed_news),
            'avg_sentiment_score': avg_score,
            'avg_confidence': avg_confidence,
            'positive_count': positive_count,
            'neutral_count': neutral_count,
            'negative_count': negative_count,
            'recommendation': recommendation,
            'news_details': analyzed_news,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def analyze_stock_list(self, stocks_dict, delay=1):
        """Analyze multiple stocks."""
        results = []
        
        print(f"\n{'='*70}")
        print(f"Analyzing {len(stocks_dict)} stocks from today's news...")
        print(f"{'='*70}\n")
        
        for ticker, company_name in tqdm(stocks_dict.items(), desc="Processing stocks"):
            result = self.analyze_stock_news(ticker, company_name)
            if result:
                results.append(result)
            time.sleep(delay)
        
        return pd.DataFrame(results)
    
    def generate_daily_report(self, results_df, save_path=None):
        """Generate a comprehensive daily report with tables."""
        if results_df.empty:
            print("No results to report.")
            return
        
        print("\n" + "="*110)
        print(f"{'DAILY STOCK ANALYSIS REPORT':^110}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^110}")
        print("="*110)
        
        # Summary statistics
        total = len(results_df)
        buy_count = len(results_df[results_df['recommendation'] == 'BUY'])
        hold_count = len(results_df[results_df['recommendation'] == 'HOLD'])
        sell_count = len(results_df[results_df['recommendation'] == 'SELL'])
        
        print(f"\nTotal Stocks Analyzed: {total}")
        print(f"📈 BUY: {buy_count} ({buy_count/total*100:.1f}%)")
        print(f"⏸️  HOLD: {hold_count} ({hold_count/total*100:.1f}%)")
        print(f"📉 SELL: {sell_count} ({sell_count/total*100:.1f}%)")
        
        # ==================== COMPLETE TABLE ====================
        print("\n" + "="*110)
        print(f"{'COMPLETE ANALYSIS TABLE':^110}")
        print("="*110)
        print(f"{'Ticker':<8} {'Company':<18} {'Action':<8} {'Score':<8} {'Conf':<8} {'+':<4} {'=':<4} {'-':<4} {'Articles':<8}")
        print("-"*110)
        
        action_order = {'BUY': 1, 'HOLD': 2, 'SELL': 3}
        df_sorted = results_df.sort_values(
            by=['recommendation', 'avg_sentiment_score'],
            key=lambda x: x.map(action_order) if x.name == 'recommendation' else x,
            ascending=[True, False]
        )
        
        for _, row in df_sorted.iterrows():
            emoji = {'BUY': '📈', 'HOLD': '⏸️', 'SELL': '📉'}[row['recommendation']]
            action_display = f"{emoji} {row['recommendation']}"
            
            print(f"{row['ticker']:<8} {row['company_name']:<18} {action_display:<9} "
                  f"{row['avg_sentiment_score']:+.3f}    {row['avg_confidence']:.1%}    "
                  f"{row['positive_count']:<4} {row['neutral_count']:<4} {row['negative_count']:<4} "
                  f"{row['total_news']:<8}")
        
        print("="*110)
        
        # ==================== BUY TABLE ====================
        print("\n" + "="*110)
        print(f"📈 STOCKS TO BUY TODAY ({buy_count} opportunities)")
        print("="*110)
        
        top_buys = results_df[results_df['recommendation'] == 'BUY'].sort_values(
            'avg_sentiment_score', ascending=False
        )
        
        if len(top_buys) > 0:
            print(f"{'Rank':<6} {'Ticker':<8} {'Company':<18} {'Score':<8} {'Confidence':<12} {'News Breakdown':<15}")
            print("-"*110)
            
            for rank, (_, row) in enumerate(top_buys.iterrows(), 1):
                news_breakdown = f"{row['positive_count']}+ / {row['neutral_count']}= / {row['negative_count']}-"
                print(f"{rank:<6} {row['ticker']:<8} {row['company_name']:<18} "
                      f"{row['avg_sentiment_score']:+.3f}    {row['avg_confidence']:.1%}        "
                      f"{news_breakdown:<15}")
        else:
            print("No BUY opportunities found. Market sentiment is cautious or neutral.")
        
        print("="*110)
        
        # ==================== HOLD TABLE ====================
        print("\n" + "="*110)
        print(f"⏸️  STOCKS TO HOLD/MONITOR ({hold_count} stocks)")
        print("="*110)
        
        holds = results_df[results_df['recommendation'] == 'HOLD'].sort_values(
            'avg_sentiment_score', ascending=False
        )
        
        if len(holds) > 0:
            print(f"{'Ticker':<8} {'Company':<18} {'Score':<8} {'Confidence':<12} {'Status':<20}")
            print("-"*110)
            
            for _, row in holds.iterrows():
                if row['avg_sentiment_score'] > 0.1:
                    status = "Slightly Positive"
                elif row['avg_sentiment_score'] < -0.1:
                    status = "Slightly Negative"
                else:
                    status = "Neutral"
                
                print(f"{row['ticker']:<8} {row['company_name']:<18} "
                      f"{row['avg_sentiment_score']:+.3f}    {row['avg_confidence']:.1%}        "
                      f"{status:<20}")
        else:
            print("No HOLD recommendations.")
        
        print("="*110)
        
        # ==================== SELL TABLE ====================
        print("\n" + "="*110)
        print(f"📉 STOCKS TO AVOID/SELL TODAY ({sell_count} warnings)")
        print("="*110)
        
        top_sells = results_df[results_df['recommendation'] == 'SELL'].sort_values(
            'avg_sentiment_score', ascending=True
        )
        
        if len(top_sells) > 0:
            print(f"{'Rank':<6} {'Ticker':<8} {'Company':<18} {'Score':<8} {'Confidence':<12} {'News Breakdown':<15}")
            print("-"*110)
            
            for rank, (_, row) in enumerate(top_sells.iterrows(), 1):
                news_breakdown = f"{row['positive_count']}+ / {row['neutral_count']}= / {row['negative_count']}-"
                print(f"{rank:<6} {row['ticker']:<8} {row['company_name']:<18} "
                      f"{row['avg_sentiment_score']:+.3f}    {row['avg_confidence']:.1%}        "
                      f"{news_breakdown:<15}")
        else:
            print("No SELL signals detected. No immediate risks identified.")
        
        print("="*110)
        
        # Save report
        if save_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = f"{save_path}_summary_{timestamp}.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"\n✓ Report saved to: {csv_path}")
        
        print("\n" + "="*110)


def main():
    analyzer = RealTimeStockNewsAnalyzer()
    
    stocks = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google',
        'AMZN': 'Amazon',
        'TSLA': 'Tesla',
        'NVDA': 'Nvidia',
        'META': 'Meta',
        'NFLX': 'Netflix',
        'AMD': 'AMD',
        'INTC': 'Intel',
    }
    
    print("\n=== Single Stock Analysis ===")
    single_result = analyzer.analyze_stock_news('NVDA', 'Nvidia', max_news=10)
    
    if single_result:
        print(f"\n{single_result['ticker']} Analysis:")
        print(f"  Recommendation: {single_result['recommendation']}")
        print(f"  Avg Sentiment: {single_result['avg_sentiment_score']:.3f}")
        print(f"  Confidence: {single_result['avg_confidence']:.2%}")
        print(f"\n  Latest News:")
        for news in single_result['news_details'][:3]:
            print(f"    • {news['headline']}")
            print(f"      Sentiment: {news['sentiment']} ({news['confidence']:.2%})")
            print(f"      Source: {news['source']}")
    
    print("\n\n=== Batch Analysis ===")
    results_df = analyzer.analyze_stock_list(stocks, delay=1)
    
    analyzer.generate_daily_report(results_df, save_path='../results/daily_stock_report')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    detailed_records = []
    for _, row in results_df.iterrows():
        for news in row['news_details']:
            detailed_records.append({
                'ticker': row['ticker'],
                'company': row['company_name'],
                'recommendation': row['recommendation'],
                'headline': news['headline'],
                'sentiment': news['sentiment'],
                'confidence': news['confidence'],
                'source': news['source'],
                'link': news['link'],
                'published': news['published']
            })
    
    detailed_df = pd.DataFrame(detailed_records)
    detailed_path = f"../results/daily_news_details_{timestamp}.csv"
    detailed_df.to_csv(detailed_path, index=False)
    print(f"✓ Detailed news saved to: {detailed_path}")


if __name__ == '__main__':
    main()