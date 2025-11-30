"""
Indian Stock News Analyzer with Live Prices and Email Reports
Complete version with full console report + beautiful HTML email + CSV attachment

Usage: python indian_stock_scanner_complete.py
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
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
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
    """Fetch real-time stock price from Yahoo Finance."""
    try:
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
    
    # Economic Times RSS
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
    
    print("Loading sentiment model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_finetuned_model('../models/finbert_sentiment_model')
    model.to(device)
    print("✓ Model loaded!\n")
    
    results = []
    
    print(f"Fetching prices and analyzing news for {len(stock_dict)} Indian stocks...\n")
    
    for ticker, company in tqdm(stock_dict.items(), desc="Processing"):
        price_data = get_stock_price(ticker)
        news_items = get_indian_stock_news(ticker, company)
        
        if not news_items or len(news_items) < 2:
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
        
        avg_weighted = sum(s['weighted'] for s in sentiments) / len(sentiments)
        avg_confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)
        
        positive_count = sum(1 for s in sentiments if s['score'] > 0)
        negative_count = sum(1 for s in sentiments if s['score'] < 0)
        neutral_count = sum(1 for s in sentiments if s['score'] == 0)
        
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


def generate_html_email(df):
    """Generate comprehensive HTML email report."""
    
    if df.empty:
        return "<p>No stocks analyzed.</p>"
    
    market_statuses = df['market_status'].value_counts()
    if 'REGULAR' in market_statuses.index or 'PRE' in market_statuses.index:
        market_status = "🟢 OPEN"
        status_color = "#10b981"
    else:
        market_status = "🔴 CLOSED"
        status_color = "#ef4444"
    
    buys = df[df['action'] == 'BUY'].sort_values('avg_score', ascending=False)
    holds = df[df['action'] == 'HOLD'].sort_values('avg_score', ascending=False)
    sells = df[df['action'] == 'SELL'].sort_values('avg_score')
    
    total = len(df)
    gainers = df[df['change_percent'] > 0]
    losers = df[df['change_percent'] < 0]
    avg_score = df['avg_score'].mean()
    
    if avg_score > 0.2:
        market_mood = "BULLISH"
        mood_color = "#10b981"
    elif avg_score < -0.2:
        market_mood = "BEARISH"
        mood_color = "#ef4444"
    else:
        market_mood = "NEUTRAL"
        mood_color = "#f59e0b"
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; padding: 20px; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 40px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 32px; }}
        .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
        .status-bar {{ background: white; padding: 20px; display: flex; justify-content: space-around; 
                       border-bottom: 3px solid #f0f0f0; }}
        .status-item {{ text-align: center; padding: 15px; }}
        .status-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .status-value {{ font-size: 28px; font-weight: bold; margin-top: 5px; }}
        .section {{ padding: 30px; border-bottom: 1px solid #f0f0f0; }}
        .section-title {{ font-size: 24px; font-weight: 600; margin-bottom: 20px; color: #333; 
                         border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th {{ background: #f8f9fa; padding: 14px; text-align: left; font-weight: 600; 
              color: #495057; border-bottom: 2px solid #dee2e6; font-size: 13px; }}
        td {{ padding: 14px; border-bottom: 1px solid #f0f0f0; font-size: 14px; }}
        tr:hover {{ background-color: #f8f9ff; }}
        .badge {{ padding: 5px 12px; border-radius: 20px; font-size: 11px; font-weight: 600; 
                  text-transform: uppercase; display: inline-block; }}
        .badge-buy {{ background: #d1fae5; color: #065f46; }}
        .badge-hold {{ background: #fef3c7; color: #92400e; }}
        .badge-sell {{ background: #fee2e2; color: #991b1b; }}
        .price-positive {{ color: #10b981; font-weight: 600; }}
        .price-negative {{ color: #ef4444; font-weight: 600; }}
        .ticker-col {{ font-weight: 600; color: #667eea; }}
        .top-picks {{ background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%); 
                      padding: 30px; margin: 0; }}
        .top-pick-item {{ background: white; padding: 20px; margin-bottom: 15px; 
                         border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; 
                        padding: 20px 0; }}
        .summary-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .footer {{ text-align: center; padding: 30px; color: #666; font-size: 12px; 
                   background: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Indian Stock Market Analysis Report</h1>
            <p>{datetime.now().strftime('%A, %B %d, %Y - %I:%M %p IST')}</p>
        </div>
        
        <div class="status-bar">
            <div class="status-item">
                <div class="status-label">Market Status</div>
                <div class="status-value" style="color: {status_color};">{market_status.replace('🟢 ', '').replace('🔴 ', '')}</div>
            </div>
            <div class="status-item">
                <div class="status-label">Stocks Analyzed</div>
                <div class="status-value">{total}</div>
            </div>
            <div class="status-item">
                <div class="status-label">Market Mood</div>
                <div class="status-value" style="color: {mood_color};">{market_mood}</div>
            </div>
            <div class="status-item">
                <div class="status-label">Sentiment Score</div>
                <div class="status-value" style="color: {mood_color};">{avg_score:+.3f}</div>
            </div>
        </div>"""
    
    # Top 3 Opportunities
    if len(buys) > 0:
        html += f"""
        <div class="top-picks">
            <h2 style="margin: 0 0 20px 0; color: #333; font-size: 26px;">🏆 Top 3 Investment Opportunities Today</h2>"""
        
        for i, (_, row) in enumerate(buys.head(3).iterrows(), 1):
            price_display = f"₹{row['price']:,.2f}" if row['price'] != 'N/A' else 'N/A'
            change_color = "#10b981" if row['change_percent'] >= 0 else "#ef4444"
            change_display = f"{row['change_percent']:+.2f}%" if row['price'] != 'N/A' else 'N/A'
            
            html += f"""
            <div class="top-pick-item">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div>
                        <div style="font-size: 20px; font-weight: 600; color: #667eea;">#{i} {row['ticker']}</div>
                        <div style="color: #666; margin-top: 5px;">{row['company']}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 24px; font-weight: 600;">{price_display}</div>
                        <div style="color: {change_color}; font-size: 16px; font-weight: 600;">{change_display}</div>
                    </div>
                </div>
                <div style="padding-top: 12px; border-top: 1px solid #e0e0e0;">
                    <div style="display: flex; justify-content: space-between; font-size: 13px; color: #666;">
                        <span>Sentiment: <strong style="color: #10b981;">{row['avg_score']:+.3f}</strong></span>
                        <span>Confidence: <strong>{row['confidence']:.1%}</strong></span>
                        <span>News: <strong>{int(row['positive_count'])}+</strong> <strong>{int(row['neutral_count'])}</strong>= <strong>{int(row['negative_count'])}</strong>-</span>
                    </div>
                    <div style="margin-top: 10px; font-size: 13px; color: #666; font-style: italic;">
                        📰 {row['top_headline'][:120]}...
                    </div>
                </div>
            </div>"""
        
        html += "</div>"
    
    # BUY Recommendations
    if len(buys) > 0:
        html += f"""
        <div class="section">
            <div class="section-title">📈 BUY Recommendations ({len(buys)} stocks)</div>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Ticker</th>
                        <th>Company</th>
                        <th style="text-align: right;">Price (₹)</th>
                        <th style="text-align: right;">Change</th>
                        <th style="text-align: center;">Score</th>
                        <th style="text-align: center;">Confidence</th>
                        <th style="text-align: center;">News</th>
                    </tr>
                </thead>
                <tbody>"""
        
        for rank, (_, row) in enumerate(buys.iterrows(), 1):
            price_display = f"₹{row['price']:,.2f}" if row['price'] != 'N/A' else 'N/A'
            change_class = 'price-positive' if row['change_percent'] >= 0 else 'price-negative'
            change_display = f"{row['change_percent']:+.2f}%" if row['price'] != 'N/A' else 'N/A'
            
            html += f"""
                <tr>
                    <td style="text-align: center; font-weight: 600;">#{rank}</td>
                    <td class="ticker-col">{row['ticker']}</td>
                    <td>{row['company'][:32]}</td>
                    <td style="text-align: right; font-weight: 600;">{price_display}</td>
                    <td class="{change_class}" style="text-align: right;">{change_display}</td>
                    <td style="text-align: center; font-weight: 600; color: #10b981;">{row['avg_score']:+.3f}</td>
                    <td style="text-align: center;">{row['confidence']:.1%}</td>
                    <td style="text-align: center;">{int(row['positive_count'])}+ {int(row['neutral_count'])}= {int(row['negative_count'])}-</td>
                </tr>"""
        
        html += """
                </tbody>
            </table>
        </div>"""
    
    # HOLD Stocks
    if len(holds) > 0:
        html += f"""
        <div class="section">
            <div class="section-title">⏸️ HOLD/Monitor Stocks ({len(holds)} stocks)</div>
            <table>
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Company</th>
                        <th style="text-align: right;">Price (₹)</th>
                        <th style="text-align: right;">Change</th>
                        <th style="text-align: center;">Score</th>
                    </tr>
                </thead>
                <tbody>"""
        
        for _, row in holds.iterrows():
            price_display = f"₹{row['price']:,.2f}" if row['price'] != 'N/A' else 'N/A'
            change_class = 'price-positive' if row['change_percent'] >= 0 else 'price-negative'
            change_display = f"{row['change_percent']:+.2f}%" if row['price'] != 'N/A' else 'N/A'
            
            html += f"""
                <tr>
                    <td class="ticker-col">{row['ticker']}</td>
                    <td>{row['company'][:32]}</td>
                    <td style="text-align: right; font-weight: 600;">{price_display}</td>
                    <td class="{change_class}" style="text-align: right;">{change_display}</td>
                    <td style="text-align: center; font-weight: 600;">{row['avg_score']:+.3f}</td>
                </tr>"""
        
        html += """
                </tbody>
            </table>
        </div>"""
    
    # SELL Warnings
    if len(sells) > 0:
        html += f"""
        <div class="section">
            <div class="section-title">📉 SELL/Avoid Warnings ({len(sells)} stocks)</div>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Ticker</th>
                        <th>Company</th>
                        <th style="text-align: right;">Price (₹)</th>
                        <th style="text-align: right;">Change</th>
                        <th style="text-align: center;">Score</th>
                        <th style="text-align: center;">Confidence</th>
                    </tr>
                </thead>
                <tbody>"""
        
        for rank, (_, row) in enumerate(sells.iterrows(), 1):
            price_display = f"₹{row['price']:,.2f}" if row['price'] != 'N/A' else 'N/A'
            change_class = 'price-positive' if row['change_percent'] >= 0 else 'price-negative'
            change_display = f"{row['change_percent']:+.2f}%" if row['price'] != 'N/A' else 'N/A'
            
            html += f"""
                <tr>
                    <td style="text-align: center; font-weight: 600;">#{rank}</td>
                    <td class="ticker-col">{row['ticker']}</td>
                    <td>{row['company'][:32]}</td>
                    <td style="text-align: right; font-weight: 600;">{price_display}</td>
                    <td class="{change_class}" style="text-align: right;">{change_display}</td>
                    <td style="text-align: center; font-weight: 600; color: #ef4444;">{row['avg_score']:+.3f}</td>
                    <td style="text-align: center;">{row['confidence']:.1%}</td>
                </tr>"""
        
        html += """
                </tbody>
            </table>
        </div>"""
    
    # Summary Statistics
    top_gainer_text = "N/A"
    top_loser_text = "N/A"
    
    if len(gainers) > 0:
        top_gainer = gainers.nlargest(1, 'change_percent').iloc[0]
        top_gainer_text = f"{top_gainer['ticker']} (+{top_gainer['change_percent']:.2f}%)"
    
    if len(losers) > 0:
        top_loser = losers.nsmallest(1, 'change_percent').iloc[0]
        top_loser_text = f"{top_loser['ticker']} ({top_loser['change_percent']:.2f}%)"
    
    html += f"""
        <div class="section">
            <div class="section-title">📊 Summary Statistics</div>
            <div class="summary-grid">
                <div class="summary-card">
                    <div style="font-size: 12px; color: #666; margin-top: 8px;">{market_mood}</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>⚠️ Disclaimer:</strong> This analysis is for informational purposes only and should not be considered as financial advice. 
            Always conduct your own research and consult with a qualified financial advisor before making investment decisions.</p>
            <p style="margin-top: 15px;">Generated by Indian Stock News Analyzer | {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}</p>
            <p style="margin-top: 10px; font-size: 11px; color: #999;">💡 TIP: Run during market hours (9:15 AM - 3:30 PM IST) for live prices!</p>
        </div>
    </div>
</body>
</html>"""
    
    return html


def send_email_report(df, recipients, csv_file, sender_email, sender_password):
    """Send email with HTML report and CSV attachment."""
    
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = sender_email
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = f"📊 Indian Stock Market Analysis - {datetime.now().strftime('%B %d, %Y')}"
        
        # Generate and attach HTML content
        html_content = generate_html_email(df)
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        # Attach CSV file
        if csv_file and os.path.exists(csv_file):
            with open(csv_file, 'rb') as file:
                attachment = MIMEBase('application', 'octet-stream')
                attachment.set_payload(file.read())
                encoders.encode_base64(attachment)
                attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename={os.path.basename(csv_file)}'
                )
                msg.attach(attachment)
        
        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipients, msg.as_string())
        server.quit()
        
        return True, "Email sent successfully!"
        
    except Exception as e:
        return False, f"Error: {str(e)}"


def print_indian_stock_report(df):
    """Print comprehensive console report with prices."""
    
    print("\n" + "="*130)
    print(f"{'INDIAN STOCK MARKET ANALYSIS WITH LIVE PRICES (NSE)':^130}")
    print(f"{datetime.now().strftime('%A, %B %d, %Y - %I:%M %p IST'):^130}")
    print("="*130)
    
    if df.empty:
        print("\n⚠ No stocks analyzed. Please check your internet connection.")
        return
    
    # Market status
    market_statuses = df['market_status'].value_counts()
    if 'REGULAR' in market_statuses.index or 'PRE' in market_statuses.index or 'POST' in market_statuses.index:
        print(f"\n🟢 Market Status: OPEN")
    else:
        print(f"\n🔴 Market Status: CLOSED (Showing last traded prices)")
    
    # Complete table
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
    
    # BUY recommendations
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
    
    # HOLD table
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
    
    # SELL table
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
    
    # Summary
    total = len(df)
    print(f"\n{'='*130}")
    print(f"{'SUMMARY STATISTICS':^130}")
    print("="*130)
    print(f"Total Stocks Analyzed: {total}")
    print(f"  📈 BUY:  {len(buys):<3} ({len(buys)/total*100:>5.1f}%)")
    print(f"  ⏸️  HOLD: {len(holds):<3} ({len(holds)/total*100:>5.1f}%)")
    print(f"  📉 SELL: {len(sells):<3} ({len(sells)/total*100:>5.1f}%)")
    
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
    """Main function with email integration."""
    
    print("\n" + "="*130)
    print(f"{'INDIAN STOCK NEWS ANALYZER WITH LIVE PRICES (NSE)':^130}")
    print("="*130)
    
    print("\nSelect stocks to analyze:")
    print("1. Top 10 Stocks (Quick scan - ~30 seconds)")
    print("2. Top 25 Stocks (Medium scan - ~2 minutes)")
    print("3. All 50 Stocks (Full scan - ~5 minutes)")
    print("4. Custom selection (enter stock tickers)")
    
    choice = input("\nEnter choice (1-4) [default: 1]: ").strip() or "1"
    
    # Email setup
    send_email = input("\n📧 Send report via email? (y/n) [default: n]: ").strip().lower()
    recipient_emails = []
    sender_email = None
    sender_password = None
    
    if send_email == 'y':
        emails_input = input("Enter recipient email(s) (comma-separated): ").strip()
        recipient_emails = [email.strip() for email in emails_input.split(',') if email.strip()]
        
        print("\n" + "="*100)
        print("EMAIL CONFIGURATION")
        print("="*100)
        print("\n📝 To send emails via Gmail:")
        print("1. Go to https://myaccount.google.com/security")
        print("2. Enable 2-Step Verification")
        print("3. Search for 'App Passwords'")
        print("4. Create password for 'Mail' app")
        print("5. Copy the 16-character password")
        print("="*100)
        
        sender_email = input("\nYour Gmail address: ").strip()
        sender_password = input("Your Gmail App Password: ").strip().replace(' ', '')
    
    # Select stocks
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
    
    # Analyze stocks
    results_df = analyze_indian_stocks(stocks)
    
    # Print console report
    print_indian_stock_report(results_df)
    
    # Save CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('../results', exist_ok=True)
    filename = f"../results/indian_stocks_with_prices_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    print(f"\n✓ Full results saved to: {filename}")
    
    # Send email
    if send_email == 'y' and recipient_emails:
        print("\n" + "="*100)
        print("📤 Sending Email Report...")
        print("="*100)
        
        success, message = send_email_report(
            results_df, 
            recipient_emails, 
            filename, 
            sender_email, 
            sender_password
        )
        
        if success:
            print(f"✅ {message}")
            print(f"📧 Beautiful HTML report sent to: {', '.join(recipient_emails)}")
            print(f"📎 CSV file attached: {os.path.basename(filename)}")
        else:
            print(f"❌ {message}")
    
    # Top opportunities
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