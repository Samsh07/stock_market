"""
Indian Stock Analyzer Web Application
Web interface for stock analysis with email reports

Deploy to Render: https://render.com/docs/deploy-node-express
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os
import sys
import torch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from sentiment_analyzer import load_finetuned_model, predict_sentiment
from email_report_sender import EmailReportSender, load_email_config

# Import stock analyzer functions
try:
    from indian_stock_analyzer import (
        INDIAN_STOCKS, get_stock_price, get_indian_stock_news, 
        analyze_indian_stocks
    )
except ImportError:
    # Fallback if indian_stock_analyzer is not available
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
    
    # Placeholder functions if indian_stock_analyzer is not available
    def get_stock_price(ticker):
        try:
            # Simple fallback - would need proper implementation
            return {
                'price': np.random.uniform(100, 3000),
                'change': np.random.uniform(-50, 50),
                'change_percent': np.random.uniform(-2, 2),
                'success': True
            }
        except:
            return {'success': False, 'price': 0, 'change': 0, 'change_percent': 0}
    
    def get_indian_stock_news(ticker, company_name):
        # Placeholder function
        return [
            {
                'ticker': ticker,
                'headline': f"Sample news for {company_name}",
                'link': "#",
                'published': datetime.now().strftime('%Y-%m-%d'),
                'source': 'Sample'
            }
        ]
    
    def analyze_indian_stocks(stock_dict, max_stocks=None):
        # Placeholder function
        if max_stocks:
            stock_dict = dict(list(stock_dict.items())[:max_stocks])
        
        results = []
        for ticker, company in stock_dict.items():
            price_data = get_stock_price(ticker)
            news_items = get_indian_stock_news(ticker, company)
            
            # Generate random sentiment scores
            avg_score = np.random.uniform(-0.5, 0.5)
            confidence = np.random.uniform(0.7, 0.95)
            
            if avg_score > 0.3 and confidence > 0.7:
                action = 'BUY'
            elif avg_score < -0.3 and confidence > 0.7:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            positive_count = np.random.randint(0, 5)
            negative_count = np.random.randint(0, 5)
            neutral_count = 5 - positive_count - negative_count
            
            results.append({
                'ticker': ticker,
                'company': company,
                'price': price_data['price'] if price_data['success'] else 'N/A',
                'change': price_data['change'] if price_data['success'] else 0,
                'change_percent': price_data['change_percent'] if price_data['success'] else 0,
                'market_status': 'CLOSED',
                'avg_score': avg_score,
                'confidence': confidence,
                'action': action,
                'positive_count': positive_count,
                'neutral_count': neutral_count,
                'negative_count': negative_count,
                'news_count': len(news_items),
                'top_headline': news_items[0]['headline'] if news_items else 'N/A'
            })
        
        return pd.DataFrame(results)

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
    }
    .buy-badge {
        background-color: #d4edda;
        color: #155724;
    }
    .hold-badge {
        background-color: #fff3cd;
        color: #856404;
    }
    .sell-badge {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load sentiment analysis model."""
    with st.spinner("Loading sentiment analysis model..."):
        try:
            tokenizer, model = load_finetuned_model('models/finbert_sentiment_model')
            return tokenizer, model, True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None, False

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Indian Stock Market Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    tokenizer, model, model_loaded = load_models()
    
    if not model_loaded:
        st.error("Model could not be loaded. Please check your model files.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Single Stock Analysis", "Batch Analysis", "Custom Text Analysis"])
    
    if page == "Single Stock Analysis":
        single_stock_page(tokenizer, model)
    elif page == "Batch Analysis":
        batch_analysis_page(tokenizer, model)
    else:
        custom_text_page(tokenizer, model)

def single_stock_page(tokenizer, model):
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

def batch_analysis_page(tokenizer, model):
    st.header("Batch Analysis")
    
    # Stock selection
    st.subheader("Select Stocks to Analyze")
    
    stock_options = {
        "Top 10 Stocks": list(INDIAN_STOCKS.items())[:10],
        "Top 25 Stocks": list(INDIAN_STOCKS.items())[:25],
        "Banking Stocks": [(k, v) for k, v in INDIAN_STOCKS.items() if 'BANK' in k or k in ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK']],
        "IT Stocks": [(k, v) for k, v in INDIAN_STOCKS.items() if k in ['INFY', 'TCS', 'WIPRO', 'HCLTECH', 'TECHM', 'MPHASIS', 'COFORGE', 'LTTS', 'TATAELXSI']],
        "Pharma Stocks": [(k, v) for k, v in INDIAN_STOCKS.items() if k in ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'LUPIN', 'AURBINDO', 'CADILAHC', 'BIOCON', 'GLENMARK', 'PIIND']],
        "Custom Selection": None
    }
    
    selected_option = st.selectbox("Choose a stock group", list(stock_options.keys()))
    
    if selected_option == "Custom Selection":
        custom_stocks = st.text_area("Enter tickers (comma-separated)", "RELIANCE, INFY, TCS")
        selected_stocks = []
        if custom_stocks:
            for ticker in custom_stocks.split(','):
                ticker = ticker.strip().upper()
                if ticker in INDIAN_STOCKS:
                    selected_stocks.append((ticker, INDIAN_STOCKS[ticker]))
                else:
                    st.warning(f"⚠ Warning: {ticker} not found")
    else:
        selected_stocks = stock_options[selected_option]
    
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
                    
                    # Email section
                    st.subheader("Email Report")
                    recipient_email = st.text_input("Recipient Email")
                    
                    if st.button("Send Email Report"):
                        if recipient_email:
                            # Load email config
                            sender_email, sender_password = load_email_config()
                            
                            if not sender_email:
                                st.error("Email not configured. Please set up email credentials first.")
                            else:
                                with st.spinner("Sending email..."):
                                    try:
                                        # Save CSV temporarily
                                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                        filename = f"temp_stock_analysis_{timestamp}.csv"
                                        results_df.to_csv(filename, index=False)
                                        
                                        # Send email
                                        email_sender = EmailReportSender(sender_email, sender_password)
                                        success = email_sender.send_report(
                                            recipient_email, 
                                            results_df, 
                                            filename
                                        )
                                        
                                        # Clean up temp file
                                        if os.path.exists(filename):
                                            os.remove(filename)
                                        
                                        if success:
                                            st.success("Email sent successfully!")
                                        else:
                                            st.error("Failed to send email.")
                                    except Exception as e:
                                        st.error(f"Error sending email: {str(e)}")
                        else:
                            st.warning("Please enter a recipient email.")
                    else:
                            st.info("No stocks selected for analysis.")
                except Exception as e:
                            st.error(f"Error analyzing stocks: {str(e)}")

def custom_text_page(tokenizer, model):
    st.header("Custom Text Analysis")
    
    # Text input
    text_input = st.text_area("Enter financial news text to analyze", height=150)
    
    if st.button("Analyze Sentiment", type="primary") and text_input:
        with st.spinner("Analyzing sentiment..."):
            try:
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

def display_results(results_df):
    """Display analysis results in tables and charts."""
    
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
        lambda row: f"🟢 {row['change_percent']:+.2f}%" if row['change_percent'] >= 0 else f"🔴 {row['change_percent']:+.2f}%",
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
    
    # Style the dataframe
    styler = display_df.style.apply(
        lambda row: [
            'background-color: #d4edda' if row['Action'] == '📈 BUY' else (
                'background-color: #f8d7da' if row['Action'] == '📉 SELL' else 'background-color: #fff3cd'
            )
        ],
        axis=1
    ).set_properties(**{
        'text-align': 'left',
        'border-collapse': 'collapse',
        'width': '100%'
    }).hide_index()
    
    st.dataframe(styler, use_container_width=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution chart
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
        # Action distribution pie chart
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

if __name__ == '__main__':
    main()