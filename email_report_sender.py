"""
Indian Stock News Analyzer with Live Prices and Email Reports
Complete standalone script - Just run: python stock_analyzer_with_email.py
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
    """Generate beautiful HTML email report."""
    
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
    holds = df[df['action'] == 'HOLD']
    sells = df[df['action'] == 'SELL']
    
    total = len(df)
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
        body {{ font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 30px; border-radius: 10px; text-align: center; }}
        .status-bar {{ background: white; padding: 20px; border-radius: 8px; 
                       margin: 20px 0; display: flex; justify-content: space-around; }}
        .status-item {{ text-align: center; }}
        .status-value {{ font-size: 24px; font-weight: bold; }}
        .section {{ background: white; padding: 25px; border-radius: 8px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th {{ background: #f8f9fa; padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6; }}
        td {{ padding: 12px; border-bottom: 1px solid #f0f0f0; }}
        .badge {{ padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }}
        .badge-buy {{ background: #d1fae5; color: #065f46; }}
        .badge-hold {{ background: #fef3c7; color: #92400e; }}
        .badge-sell {{ background: #fee2e2; color: #991b1b; }}
        .price-positive {{ color: #10b981; font-weight: 600; }}
        .price-negative {{ color: #ef4444; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Indian Stock Market Analysis</h1>
            <p>{datetime.now().strftime('%A, %B %d, %Y - %I:%M %p IST')}</p>
        </div>
        
        <div class="status-bar">
            <div class="status-item">
                <div>Market Status</div>
                <div class="status-value" style="color: {status_color};">{market_status}</div>
            </div>
            <div class="status-item">
                <div>Stocks Analyzed</div>
                <div class="status-value">{total}</div>
            </div>
            <div class="status-item">
                <div>Market Mood</div>
                <div class="status-value" style="color: {mood_color};">{market_mood}</div>
            </div>
            <div class="status-item">
                <div>Sentiment</div>
                <div class="status-value" style="color: {mood_color};">{avg_score:+.3f}</div>
            </div>
        </div>"""
    
    # Top picks
    if len(buys) > 0:
        html += '<div class="section"><h2>🏆 Top Investment Opportunities</h2>'
        for i, (_, row) in enumerate(buys.head(3).iterrows(), 1):
            price = f"₹{row['price']:,.2f}" if row['price'] != 'N/A' else 'N/A'
            change = f"{row['change_percent']:+.2f}%" if row['price'] != 'N/A' else 'N/A'
            html += f"""
            <div style="background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px;">
                <strong>#{i} {row['ticker']} - {row['company']}</strong><br>
                Price: {price} ({change}) | Score: {row['avg_score']:+.3f}
            </div>"""
        html += '</div>'
    
    # BUY table
    if len(buys) > 0:
        html += '<div class="section"><h2>📈 BUY Recommendations</h2><table><tr>'
        html += '<th>Ticker</th><th>Company</th><th>Price</th><th>Change</th><th>Score</th></tr>'
        for _, row in buys.iterrows():
            price = f"₹{row['price']:,.2f}" if row['price'] != 'N/A' else 'N/A'
            change_class = 'price-positive' if row['change_percent'] >= 0 else 'price-negative'
            change = f"{row['change_percent']:+.2f}%" if row['price'] != 'N/A' else 'N/A'
            html += f"""<tr>
                <td style="font-weight: bold; color: #667eea;">{row['ticker']}</td>
                <td>{row['company'][:30]}</td>
                <td style="font-weight: bold;">{price}</td>
                <td class="{change_class}">{change}</td>
                <td>{row['avg_score']:+.3f}</td>
            </tr>"""
        html += '</table></div>'
    
    # HOLD table
    if len(holds) > 0:
        html += '<div class="section"><h2>⏸️ HOLD Stocks</h2><table><tr>'
        html += '<th>Ticker</th><th>Company</th><th>Price</th><th>Change</th></tr>'
        for _, row in holds.iterrows():
            price = f"₹{row['price']:,.2f}" if row['price'] != 'N/A' else 'N/A'
            change = f"{row['change_percent']:+.2f}%" if row['price'] != 'N/A' else 'N/A'
            html += f"<tr><td>{row['ticker']}</td><td>{row['company'][:30]}</td><td>{price}</td><td>{change}</td></tr>"
        html += '</table></div>'
    
    # SELL table
    if len(sells) > 0:
        html += '<div class="section"><h2>📉 SELL Warnings</h2><table><tr>'
        html += '<th>Ticker</th><th>Company</th><th>Price</th><th>Score</th></tr>'
        for _, row in sells.iterrows():
            price = f"₹{row['price']:,.2f}" if row['price'] != 'N/A' else 'N/A'
            html += f"<tr><td>{row['ticker']}</td><td>{row['company'][:30]}</td><td>{price}</td><td>{row['avg_score']:+.3f}</td></tr>"
        html += '</table></div>'
    
    html += """
        <div style="text-align: center; padding: 20px; color: #666; font-size: 12px;">
            <p><strong>Disclaimer:</strong> This is for informational purposes only. Not financial advice.</p>
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
        msg['Subject'] = f"📊 Stock Market Analysis - {datetime.now().strftime('%B %d, %Y')}"
        
        html_content = generate_html_email(df)
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        # Attach CSV
        if csv_file and os.path.exists(csv_file):
            with open(csv_file, 'rb') as file:
                attachment = MIMEBase('application', 'octet-stream')
                attachment.set_payload(file.read())
                encoders.encode_base64(attachment)
                attachment.add_header('Content-Disposition', f'attachment; filename={os.path.basename(csv_file)}')
                msg.attach(attachment)
        
        # Send
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipients, msg.as_string())
        server.quit()
        
        return True, "Email sent successfully!"
        
    except Exception as e:
        return False, f"Error: {str(e)}"


def print_console_report(df):
    """Print report to console."""
    print("\n" + "="*100)
    print(f"{'STOCK ANALYSIS RESULTS':^100}")
    print("="*100)
    
    buys = df[df['action'] == 'BUY']
    holds = df[df['action'] == 'HOLD']
    sells = df[df['action'] == 'SELL']
    
    print(f"\n📈 BUY: {len(buys)} | ⏸️ HOLD: {len(holds)} | 📉 SELL: {len(sells)}")
    print(f"Market Sentiment: {df['avg_score'].mean():+.3f}")
    
    if len(buys) > 0:
        print("\n🏆 TOP BUY RECOMMENDATIONS:")
        for i, (_, row) in enumerate(buys.head(5).iterrows(), 1):
            print(f"{i}. {row['ticker']} - ₹{row['price']:,.2f} ({row['change_percent']:+.2f}%) - Score: {row['avg_score']:+.3f}")
    
    print("="*100)


def main():
    """Main function."""
    
    print("\n" + "="*100)
    print(f"{'INDIAN STOCK ANALYZER WITH EMAIL REPORTS':^100}")
    print("="*100)
    
    print("\nSelect stocks to analyze:")
    print("1. Top 10 Stocks (~30 seconds)")
    print("2. Top 25 Stocks (~2 minutes)")
    print("3. All 50 Stocks (~5 minutes)")
    
    choice = input("\nEnter choice (1-3) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        stocks = dict(list(INDIAN_STOCKS.items())[:10])
    elif choice == "2":
        stocks = dict(list(INDIAN_STOCKS.items())[:25])
    else:
        stocks = INDIAN_STOCKS
    
    # Email setup
    send_email = input("\n📧 Send report via email? (y/n) [default: n]: ").strip().lower()
    
    if send_email == 'y':
        print("\n" + "="*100)
        print("EMAIL SETUP")
        print("="*100)
        print("\n📝 To send emails via Gmail:")
        print("1. Go to https://myaccount.google.com/security")
        print("2. Enable 2-Step Verification")
        print("3. Search for 'App Passwords'")
        print("4. Create password for 'Mail' app")
        print("5. Use the 16-character password below")
        print("="*100)
        
        sender_email = input("\nYour Gmail address: ").strip()
        sender_password = input("Your App Password (16 chars): ").strip().replace(' ', '')
        recipients = input("Recipient email(s) (comma-separated): ").strip().split(',')
        recipients = [r.strip() for r in recipients if r.strip()]
    
    # Analyze
    print(f"\n✓ Analyzing {len(stocks)} stocks...\n")
    results_df = analyze_indian_stocks(stocks)
    
    # Save CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('../results', exist_ok=True)
    csv_file = f"../results/stocks_{timestamp}.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"\n✓ Results saved: {csv_file}")
    
    # Print console report
    print_console_report(results_df)
    
    # Send email
    if send_email == 'y' and recipients:
        print("\n📤 Sending email...")
        success, message = send_email_report(results_df, recipients, csv_file, sender_email, sender_password)
        if success:
            print(f"✅ {message}")
            print(f"📧 Report sent to: {', '.join(recipients)}")
        else:
            print(f"❌ {message}")
    
    print("\n" + "="*100)
    print("✓ Analysis complete!")
    print("="*100)


if __name__ == '__main__':
    main()