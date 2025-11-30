"""
Indian Stock Analyzer - Complete Web Application
Includes all functionality in one file for easier deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
import sys
from sentiment_analyzer import predict_sentiment
import torch
from tqdm import tqdm
import requests
import feedparser
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Include all the code from sentiment_analyzer.py here
# (Copy the entire sentiment_analyzer.py file content here)

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

# Configure page
st.set_page_config(
    page_title="Indian Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .positive { color: #1f77b4; }
    .negative { color: #d62728; }
    .neutral { color: #7f7f7f; }
    .buy-row { background-color: #d4edda; }
    .sell-row { background-color: #f8d7da; }
    .stock-table {
        border-collapse: collapse;
        width: 100%;
        margin-top: 1rem;
    }
    .stock-table th {
        background-color: #1f77b4;
        color: white;
        padding: 0.75rem;
        text-align: left;
        font-weight: bold;
    }
    .stock-table td {
        padding: 0.75rem;
        border-bottom: 1px solid #ddd;
    }
    .action-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        font-size: 0.8rem;
        text-transform: uppercase;
        display: inline-block;
    }
    .buy-badge {
        background-color: #d1fae5;
        color: #065f46;
    }
    .hold-badge {
        background-color: #fff3cd;
        color: #92400e;
    }
    .sell-badge {
        background-color: #fee2e2;
        color: #991b1b;
    }
    .price-positive { color: #10b981; font-weight: 600; }
    .price-negative { color: #ef4444; font-weight: 600; }
    .ticker-col { font-weight: 600; color: #667eea; }
    .top-picks {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        padding: 2rem;
        margin: 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .top-pick-item {
        background: white;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        font-size: 0.8rem;
        background-color: #f8f9fa;
        border-top: 1px solid #ddd;
        border-radius: 0.5rem 0.5rem 0 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load sentiment analysis model."""
    with st.spinner("Loading sentiment analysis model..."):
        # Load your fine-tuned local model
        model_path = r"C:\sentimental_model\models\finbert_sentiment_model"

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        return tokenizer, model, True

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
    
    tokenizer, model, model_loaded = load_models()
    
    if not model_loaded:
        st.error("Model could not be loaded. Please check your model files.")
        return pd.DataFrame()
    
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
        
        # Simple sentiment analysis (fallback if model not available)
        if model and tokenizer:
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
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
            except Exception as e:
                st.error(f"Error analyzing {ticker}: {str(e)}")
                results.append({
                    'ticker': ticker,
                    'company': company,
                    'price': price_data['price'] if price_data['success'] else 'N/A',
                    'change': price_data['change'] if price_data['success'] else 0,
                    'change_percent': price_data['change_percent'] if price_data['success'] else 0,
                    'market_status': price_data.get('market_status', 'UNKNOWN'),
                    'avg_score': 0,
                    'confidence': 0,
                    'action': 'HOLD',
                    'positive_count': 0,
                    'neutral_count': 0,
                    'negative_count': 0,
                    'news_count': 0,
                    'top_headline': 'Analysis failed'
                })
        
        time.sleep(0.5)
    
    return pd.DataFrame(results)

def display_results(results_df):
    """Display analysis results in tables and charts."""
    
    if results_df.empty:
        st.warning("No stocks analyzed. Please check your internet connection.")
        return
    
    # Summary metrics
    total = len(results_df)
    buys = results_df[results_df['action'] == 'BUY']
    holds = results_df[results_df['action'] == 'HOLD']
    sells = results_df[results_df['action'] == 'SELL']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Buy Recommendations", len(buys))
    with col2:
        st.metric("Hold Recommendations", len(holds))
    with col3:
        st.metric("Sell Recommendations", len(sells))
    
    # Detailed table
    st.subheader("Detailed Analysis")
    
    # Format dataframe for display
    display_df = results_df.copy()
    display_df['Price (₹)'] = display_df['price'].apply(lambda x: f"₹{x:,.2f}" if x != 'N/A' else 'N/A')
    display_df['Change'] = display_df.apply(
        lambda row: f"🟢 +{row['change_percent']:.2f}%" if row['change_percent'] >= 0 else f"🔴 {row['change_percent']:.2f}%",
        axis=1
    )
    display_df['Action'] = display_df.apply(
        lambda row: f"📈 {row['action']}" if row['action'] == 'BUY' else (
            f"📉 {row['action']}" if row['action'] == 'SELL' else f"⏸️ {row['action']}"
        ),
        axis=1
    )
    
    # Select columns to display
    columns_to_show = ['ticker', 'company', 'Price (₹)', 'Change', 'Action', 'avg_score', 'confidence']
    display_df = display_df[columns_to_show]
    display_df.columns = ['Ticker', 'Company', 'Price', 'Change', 'Action', 'Sentiment Score', 'Confidence']
    
    # Number of columns to style
    n_cols = len(display_df.columns)

    # Style dataframe
    styler = display_df.style.apply(
        lambda row: [
            'background-color: #c8e6c9; color: #000000' if row['Action'] == '📈 BUY' else
        ('background-color: #ffcdd2; color: #000000' if row['Action'] == '📉 SELL' else
         'background-color: #fff9c4; color: #000000')
    ] * n_cols,
    axis=1
    ).set_properties(**{
        'text-align': 'left',
    'border-collapse': 'collapse',
    'width': '100%',
    'font-weight': 'bold'
    })

    st.dataframe(styler, use_container_width=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        fig = px.histogram(
            results_df, 
            x="avg_score", 
            nbins=20,
            title="Distribution of Sentiment Scores",
            labels={"avg_score": "Sentiment Score", "count": "Number of Stocks"}
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
        fig.add_vline(x=0.2, line_dash="dash", line_color="green", annotation_text="Buy Threshold")
        fig.add_vline(x=-0.2, line_dash="dash", line_color="red", annotation_text="Sell Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Action Distribution")
        action_counts = results_df['action'].value_counts()
        fig = px.pie(
            names=action_counts.index,
            values=action_counts.values,
            title="Distribution of Actions",
            color_discrete_map={
                'BUY': '#28a745',
                'HOLD': '#ffc107',
                'SELL': '#dc3545'
            }
        )
        st.plotly_chart(fig, use_container_width=True)


def send_email_report(results_df, recipient_email, sender_email, sender_password):
    """Send email with HTML report and CSV attachment."""
    
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"📊 Indian Stock Analysis Report - {datetime.now().strftime('%B %d, %Y')}"
        msg['From'] = sender_email
        msg['To'] = recipient_email
        
        # Generate HTML content
        html_content = generate_html_email(results_df)
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        # Attach CSV file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = f"../results/stocks_{timestamp}.csv"
        results_df.to_csv(csv_file, index=False)
        
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
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        
        return True, "Email sent successfully!"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def generate_html_email(results_df):
    """Generate HTML email report."""
    
    if results_df.empty:
        return "<p>No stocks analyzed.</p>"
    
    # Calculate summary stats
    total = len(results_df)
    buys = results_df[results_df['action'] == 'BUY']
    holds = results_df[results_df['action'] == 'HOLD']
    sells = results_df[results_df['action'] == 'SELL']
    
    buy_count = len(buys)
    hold_count = len(holds)
    sell_count = len(sells)
    
    avg_score = results_df['avg_score'].mean()
    
    # Determine market mood
    if avg_score > 0.2:
        market_mood = "BULLISH"
        mood_color = "#10b981"
    elif avg_score < -0.2:
        market_mood = "BEARISH"
        mood_color = "#ef4444"
    else:
        market_mood = "NEUTRAL"
        mood_color = "#ffc107"
    
    # Start HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 10px 10px 0 0;
            margin: -30px -30px 30px -30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .summary {{
            display: flex;
            justify-content: space-around;
            margin: 30px 0;
        }}
        .summary-box {{
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            flex: 1;
            margin: 0 10px;
        }}
        .summary-box h2 {{
            margin: 0;
            font-size: 36px;
            color: #333;
        }}
        .summary-box p {{
            margin: 5px 0 0 0;
            color: #666;
            font-size: 14px;
        }}
        .market-mood {{
            text-align: center;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            margin: 30px 0;
        }}
        .market-mood h3 {{
            margin: 0;
            color: {mood_color};
            font-size: 24px;
        }}
        .section {{
            padding: 30px;
            border-bottom: 1px solid #f0f0f0;
            margin: 30px 0;
        }}
        .section-title {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 14px;
        }}
        th {{
            background-color: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f8f9ff;
        }}
        .badge {{
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            display: inline-block;
        }}
        .badge-buy {{
            background-color: #d1fae5;
            color: #065f46;
        }}
        .badge-hold {{
            background-color: #fff3cd;
            color: #92400e;
        }}
        .badge-sell {{
            background-color: #fee2e2;
            color: #991b1b;
        }}
        .price {{
            font-weight: bold;
            color: #333;
        }}
        .price-positive {{
            color: #10b981;
            font-weight: 600;
        }}
        .price-negative {{
            color: #ef4444;
            font-weight: 600;
        }}
        .ticker-col {{
            font-weight: 600;
            color: #667eea;
        }}
        .top-picks {{
            background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
            padding: 30px;
            margin: 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .top-pick-item {{
            background: white;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .footer {{
            text-align: center;
            padding: 30px 0;
            color: #666;
            font-size: 12px;
            background-color: #f8f9fa;
            border-top: 1px solid #ddd;
            border-radius: 0.5rem 0.5rem 0 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Indian Stock Market Analysis Report</h1>
            <p>{datetime.now().strftime('%A, %B %d, %Y - %I:%M %p IST')}</p>
        </div>
        
        <div class="summary">
            <div class="summary-box buy">
                <h2>{buy_count}</h2>
                <p>📈 STOCKS TO BUY</p>
            </div>
            <div class="summary-box hold">
                <h2>{hold_count}</h2>
                <p>⏸️ STOCKS TO HOLD</p>
            </div>
            <div class="summary-box sell">
                <h2>{sell_count}</h2>
                <p>📉 STOCKS TO AVOID</p>
            </div>
        </div>
        
        <div class="market-mood">
            <h3>Market Mood: {market_mood}</h3>
            <p>Overall Sentiment Score: {avg_score:+.3f}</p>
        </div>
        
        <div class="section">
            <div class="section-title">📈 Top Buy Recommendations</div>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Ticker</th>
                    <th>Company</th>
                    <th>Price (₹)</th>
                    <th>Change</th>
                    <th>Score</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>"""
    
    # Add rows for each stock
    for rank, (_, row) in enumerate(buys.head(3).iterrows(), 1):
        price_display = f"₹{row['price']:,.2f}" if row['price'] != 'N/A' else 'N/A'
        change_class = "price-positive" if row['change_percent'] >= 0 else "price-negative"
        change_symbol = "+" if row['change_percent'] >= 0 else ""
        change_display = f"{change_symbol}{row['change_percent']:.2f}%" if row['price'] != 'N/A' else 'N/A'
        
        html += f"""
                <tr>
                    <td style="text-align: center; font-weight: 600;">#{rank}</td>
                    <td class="ticker-col">{row['ticker']}</td>
                    <td>{row['company'][:32]}</td>
                    <td style="text-align: right; font-weight: 600;">{price_display}</td>
                    <td class="{change_class}" style="text-align: right;">{change_display}</td>
                    <td style="text-align: center; font-weight: 600; color: #10b981;">{row['avg_score']:+.3f}</td>
                    <td style="text-align: center;">{row['confidence']:.1%}</td>
                </tr>
            """
    
    html += """
            </tbody>
            </table>
        </div>
        
        <div class="section">
            <div class="section-title">⏸️ HOLD/Monitor Stocks</div>
            <table>
                <tr>
                    <th>Ticker</th>
                    <th>Company</th>
                    <th>Price (₹)</th>
                    <th>Change</th>
                    <th>Score</th>
                </tr>
            </thead>
            <tbody>"""
    
    for _, row in holds.iterrows():
        price_display = f"₹{row['price']:,.2f}" if row['price'] != 'N/A' else 'N/A'
        change_class = "price-positive" if row['change_percent'] >= 0 else "price-negative"
        change_symbol = "+" if row['change_percent'] >= 0 else ""
        change_display = f"{change_symbol}{row['change_percent']:.2f}%" if row['price'] != 'N/A' else 'N/A'
        
        html += f"""
                <tr>
                    <td class="ticker-col">{row['ticker']}</td>
                    <td>{row['company'][:32]}</td>
                    <td style="text-align: right; font-weight: 600;">{price_display}</td>
                    <td class="{change_class}" style="text-align: right;">{change_display}</td>
                    <td style="text-align: center; font-weight: 600;">{row['avg_score']:+.3f}</td>
                </tr>
            """
    
    html += """
            </tbody>
            </table>
        </div>
        
        <div class="section">
            <div class="section-title">📉 SELL/Avoid Warnings</div>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Ticker</th>
                    <th>Company</th>
                    <th>Price (₹)</th>
                    <th>Change</th>
                    <th>Score</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>"""
    
    for rank, (_, row) in enumerate(sells.head(5).iterrows(), 1):
        price_display = f"₹{row['price']:,.2f}" if row['price'] != 'N/A' else 'N/A'
        change_class = "price-positive" if row['change_percent'] >= 0 else "price-negative"
        change_symbol = "+" if row['change_percent'] >= 0 else ""
        change_display = f"{change_symbol}{row['change_percent']:.2f}%" if row['price'] != 'N/A' else 'N/A'
        
        html += f"""
                <tr>
                    <td style="text-align: center; font-weight: 600;">#{rank}</td>
                    <td class="ticker-col">{row['ticker']}</td>
                    <td>{row['company'][:32]}</td>
                    <td style="text-align: right; font-weight: 600;">{price_display}</td>
                    <td class="{change_class}" style="text-align: right;">{change_display}</td>
                    <td style="text-align: center; font-weight: 600; color: #ef4444;">{row['avg_score']:+.3f}</td>
                    <td style="text-align: center;">{row['confidence']:.1%}</td>
                </tr>
            """
    
    html += """
            </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>This report was generated using AI-powered sentiment analysis of financial news.</p>
            <p><strong>Disclaimer:</strong> This is for informational purposes only. Not financial advice. Always do your own research.</p>
            <p>© 2025 Stock Sentiment Analyzer | Powered by FinBERT</p>
        </div>
    </div>
</body>
</html>
    """
    
    return html

def main():
    """Main function with Streamlit interface."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Indian Stock Market Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Single Stock Analysis", "Batch Analysis", "Custom Text Analysis"])
    
    if page == "Single Stock Analysis":
        single_stock_page()
    elif page == "Batch Analysis":
        batch_analysis_page()
    else:
        custom_text_page()

def single_stock_page():
    """Single stock analysis page."""
    st.header("Single Stock Analysis")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input("Enter Stock Ticker", "RELIANCE", help="Enter NSE ticker without suffix (e.g., RELIANCE for NSE:RELIANCE-EQ)")
    
    with col2:
        analyze_button = st.button("Analyze", type="primary")
    
    if analyze_button and ticker:
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                # Get stock price
                price_data = get_stock_price(ticker)
                
                # Get news
                company_name = INDIAN_STOCKS.get(ticker, "Unknown Company")
                news_items = get_indian_stock_news(ticker, company_name)
                
                # Display stock info
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if price_data['success']:
                        st.metric("Current Price", f"₹{price_data['price']:,.2f}")
                    else:
                        st.metric("Price", "N/A")
                
                with col2:
                    if price_data['success']:
                        change_color = "normal" if price_data['change_percent'] >= 0 else "inverse"
                        st.metric("Change", f"{price_data['change_percent']:+.2f}%", delta=None, delta_color=change_color)
                    else:
                        st.metric("Change", "N/A")
                
                with col3:
                    if price_data['success']:
                        st.metric("Market Status", price_data.get('market_status', 'UNKNOWN'))
                    else:
                        st.metric("Market Status", "UNKNOWN")
                
                with col4:
                    st.metric("News Articles", len(news_items))
                
                # Display news headlines
                if news_items:
                    st.subheader("Recent News Headlines")
                    for i, news in enumerate(news_items[:5], 1):
                        sentiment, confidence = predict_sentiment(
                            news['headline'], model, tokenizer, device='cpu'
                        )
                        
                        sentiment_class = "positive" if sentiment == "Positive" else ("negative" if sentiment == "Negative" else "neutral")
                        
                        st.markdown(f"""
                            <div class="metric-card">
                                <p><b>{i}. {news['headline']}</b></p>
                                <p><small>Source: {news['source']} | Published: {news['published']}</small></p>
                                <p><span class="{sentiment_class}">Sentiment: {sentiment} (Confidence: {confidence:.1%})</span></p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No recent news headlines found.")
                    
            except Exception as e:
                st.error(f"Error analyzing {ticker}: {str(e)}")

def batch_analysis_page():
    """Batch analysis page with email option."""
    st.header("Batch Analysis")
    
    # Stock selection
    st.subheader("Select Stocks to Analyze")
    
    stock_options = {
        "Top 10 Stocks": list(INDIAN_STOCKS.items())[:10],
        "Top 25 Stocks": list(INDIAN_STOCKS.items())[:25],
        "All 50 Stocks": list(INDIAN_STOCKS.items())
    }
    
    selected_option = st.selectbox("Choose a stock group", list(stock_options.keys()))
    selected_stocks = stock_options[selected_option]
    
    # Email setup
    email_option = st.checkbox("Send Report via Email")
    
    if email_option:
        st.subheader("Email Configuration")
        sender_email = st.text_input("Your Gmail Address")
        sender_password = st.text_input("Your Gmail App Password", type="password")
    
    # Analyze button
    if st.button("Analyze Selected Stocks", type="primary"):
        if selected_stocks:
            with st.spinner("Analyzing stocks..."):
                try:
                    # Convert to dictionary for analyze function
                    stocks_dict = dict(selected_stocks)
                    
                    # Analyze stocks
                    results_df = analyze_indian_stocks(stocks_dict)
                    
                    # Display results
                    display_results(results_df)
                    
                    # Show download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name=f'stock_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv'
                    )
                    
                    # Send email if requested
                    if email_option and sender_email and sender_password:
                        with st.spinner("Sending email..."):
                            success, message = send_email_report(
                                results_df, 
                                sender_email, 
                                sender_email, 
                                sender_password
                            )
                            
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                except Exception as e:
                    st.error(f"Error analyzing stocks: {str(e)}")
                else:
                    st.info("No stocks selected for analysis.")

def custom_text_page():
    """Custom text analysis page."""
    st.header("Custom Text Analysis")
    
    # Text input
    text_input = st.text_area("Enter financial news text to analyze", height=150)
    
    if st.button("Analyze Sentiment", type="primary") and text_input:
        with st.spinner("Analyzing sentiment..."):
            try:
                tokenizer, model, model_loaded = load_models()
                
                if model_loaded:
                    sentiment, confidence = predict_sentiment(
                        text_input, model, tokenizer, device='cpu'
                    )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        sentiment_class = "positive" if sentiment == "Positive" else ("negative" if sentiment == "Negative" else "neutral")
                        st.markdown(f"<h2 class='{sentiment_class}'>{sentiment}</h2>", unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Sentiment gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = 0 if sentiment == "Neutral" else (1 if sentiment == "Positive" else -1),
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Sentiment Score"},
                        gauge = {
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [-1, -0.33], 'color': "lightgray"},
                                {'range': [-0.33, 0.33], 'color': "gray"},
                                {'range': [0.33, 1], 'color': "lightgray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error analyzing text: {str(e)}")
            else:
                st.error("Model not loaded. Please check your model files.")

if __name__ == '__main__':
    main()