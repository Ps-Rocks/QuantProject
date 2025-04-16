import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from datetime import datetime, timedelta
from scipy import stats
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import time
import torch
import re
from bs4 import BeautifulSoup
import PyPDF2
import io

# Load the sentiment analysis model
print("Loading sentiment analysis model...")
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Load a pre-trained model for ESG classification
print("Loading ESG classification model...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# Define ESG keywords for classification
esg_keywords = {
    'environmental': [
        'climate change', 'carbon emissions', 'renewable energy', 'pollution', 'waste management',
        'water usage', 'biodiversity', 'deforestation', 'sustainability', 'green energy',
        'environmental impact', 'carbon footprint', 'clean energy', 'recycling', 'conservation'
    ],
    'social': [
        'human rights', 'labor practices', 'employee welfare', 'diversity', 'inclusion',
        'community relations', 'customer satisfaction', 'data privacy', 'product safety', 'gender equality',
        'workplace safety', 'fair wages', 'social responsibility', 'health', 'education'
    ],
    'governance': [
        'board diversity', 'executive compensation', 'shareholder rights', 'audit committee', 'business ethics',
        'corruption', 'transparency', 'compliance', 'risk management', 'lobbying',
        'political contributions', 'tax strategy', 'whistleblower protection', 'board independence', 'accountability'
    ]
}

# Define stocks
stocks = ['MSFT', 'NVDA', 'AAPL', 'AMZN', 'GOOGL', 'TSLA', 'JNJ', 'PG', 'V', 'WMT']

def get_news_from_yahoo_finance(stocks, days_back=7):
    """Fetch news from Yahoo Finance"""
    all_news = {}
    for stock in stocks:
        print(f"Fetching Yahoo Finance news for {stock}...")
        ticker = yf.Ticker(stock)
        try:
            # Get news from Yahoo Finance
            news = ticker.news
            # Extract titles and summaries
            texts = []
            for item in news:
                title = item['content']['title']
                summary = item['content']['summary']
                full_text = f"{title}. {summary}"
                texts.append(full_text)
            all_news[stock] = texts
        except Exception as e:
            print(f"Error fetching Yahoo Finance news for {stock}: {e}")
            all_news[stock] = []
        # Respecting the API rate limits
        time.sleep(1)
    return all_news

def analyze_sentiment(news_dict):
    """Analyze sentiment for each stock's news"""
    sentiment_scores = {}
    for stock, texts in news_dict.items():
        if not texts:
            # If no news, assign neutral sentiment
            sentiment_scores[stock] = 0.75
            continue
        # Analyze sentiment for each news item
        stock_sentiments = []
        # Process in batches to avoid memory issues
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            results = sentiment_pipeline(batch)
            for result in results:
                if result['label'] == 'positive':
                    score = 0.75 + (result['score'] * 0.25)
                    print("Positive Sentiment")
                elif result['label'] == 'negative':
                    score = 0.5 + (result['score'] * 0.25)
                    print("Negative Sentiment")
                else: # NEUTRAL
                    score = 0.75
                    print("Neutral Sentiment")
                stock_sentiments.append(score)
        # Average sentiment score for the stock
        if stock_sentiments:
            sentiment_scores[stock] = sum(stock_sentiments) / len(stock_sentiments)
        else:
            sentiment_scores[stock] = 0.75
    return sentiment_scores

def predict_esg_scores(news_dict):
    """Predict ESG scores using NLP on news articles"""
    esg_scores = {}
    for stock, texts in news_dict.items():
        if not texts:
            # If no news, assign default ESG score
            esg_scores[stock] = 0.85
            continue
        # Combine all news texts
        combined_text = " ".join(texts)
        # Calculate ESG category scores based on keyword presence
        env_score = 0
        soc_score = 0
        gov_score = 0
        # Count keyword occurrences for each category
        for keyword in esg_keywords['environmental']:
            env_score += len(re.findall(r'\b' + re.escape(keyword) + r'\b', combined_text.lower()))
        for keyword in esg_keywords['social']:
            soc_score += len(re.findall(r'\b' + re.escape(keyword) + r'\b', combined_text.lower()))
        for keyword in esg_keywords['governance']:
            gov_score += len(re.findall(r'\b' + re.escape(keyword) + r'\b', combined_text.lower()))
        # Get sentiment for ESG-related sentences
        esg_sentences = []
        for text in texts:
            sentences = re.split(r'[.!?]', text)
            for sentence in sentences:
                # Check if sentence contains any ESG keyword
                contains_esg = False
                for category in esg_keywords.values():
                    for keyword in category:
                        if keyword in sentence.lower():
                            contains_esg = True
                            break
                    if contains_esg:
                        break
                if contains_esg:
                    esg_sentences.append(sentence)
        # Analyze sentiment of ESG-related sentences
        esg_sentiments = []
        if esg_sentences:
            batch_size = 8
            for i in range(0, len(esg_sentences), batch_size):
                batch = esg_sentences[i:i+batch_size]
                # Filter out empty sentences
                batch = [s for s in batch if s.strip()]
                if batch:
                    results = sentiment_pipeline(batch)
                    for result in results:
                        if result['label'] == 'POSITIVE':
                            score = 0.75 + (result['score'] * 0.25)
                        elif result['label'] == 'NEGATIVE':
                            score = 0.5 + (result['score'] * 0.25)
                        else: # NEUTRAL
                            score = 0.75
                        esg_sentiments.append(score)
        # Calculate final ESG score
        if esg_sentiments:
            sentiment_factor = sum(esg_sentiments) / len(esg_sentiments)
        else:
            sentiment_factor = 0.75 # Neutral
        # Calculate total ESG mentions
        total_mentions = env_score + soc_score + gov_score
        if total_mentions > 0:
            # Calculate weighted score based on mentions and sentiment
            # Normalize to 0.7-1.0 range
            normalized_mentions = min(total_mentions / 50, 1.0) # Cap at 50 mentions
            esg_score = 0.7 + (normalized_mentions * sentiment_factor * 0.3)
        else:
            esg_score = 0.85 # Default value
        esg_scores[stock] = esg_score
        print(f"{stock} ESG Analysis: E={env_score}, S={soc_score}, G={gov_score}, Score={esg_score:.4f}")
    return esg_scores

# Main portfolio optimization function
def optimize_portfolio():
    # Set date range for backtesting (2 years from current date)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    # Fetch historical data
    print("Downloading historical stock data...")
    stock_data = yf.download(stocks, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
    
    # Calculate daily returns
    returns = stock_data.pct_change().dropna()
    
    # Calculate conventional parameters
    # 1. Sharpe Ratio (assuming risk-free rate of 2%)
    risk_free_rate = 0.02 / 252 # Daily risk-free rate
    sharpe_ratios = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
    
    # 2. Beta (using S&P 500 as market benchmark)
    market_data = yf.download('^GSPC', start=start_date, end=end_date, auto_adjust=False)['Adj Close']
    market_returns = market_data.pct_change().dropna()
    
    # Calculate betas using a simpler approach
    betas = []
    for stock in stocks:
        # Get common dates
        common_idx = returns.index.intersection(market_returns.index)
        # Calculate beta using numpy's polyfit
        x = market_returns.loc[common_idx].values
        y = returns[stock].loc[common_idx].values
        # Reshape to ensure 1D arrays
        x = x.reshape(-1)
        y = y.reshape(-1)
        # Calculate beta as the slope of the regression line
        beta = np.polyfit(x, y, 1)[0]
        betas.append(beta)
    
    # 3. Maximum Drawdown
    max_drawdowns = []
    for stock in stocks:
        cumulative_returns = (1 + returns[stock]).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdowns.append(abs(drawdown.min()))
    
    # Get real sentiment scores using local sentiment analysis
    print("Fetching and analyzing news for sentiment scores...")
    all_news = get_news_from_yahoo_finance(stocks)
    sentiment_scores_dict = analyze_sentiment(all_news)
    sentiment_scores = np.array([sentiment_scores_dict[stock] for stock in stocks])
    
    # Get ESG scores using NLP on news articles
    print("Predicting ESG scores from news articles...")
    esg_scores_dict = predict_esg_scores(all_news)
    esg_scores = np.array([esg_scores_dict[stock] for stock in stocks])
    
    # Calculate volatility
    volatility = returns.std().values
    
    # Composite score calculation with conventional parameters
    composite_scores = (
        sentiment_scores +
        (1 / volatility) +
        esg_scores +
        sharpe_ratios.values +
        (1 / np.array(betas)) +
        (1 / np.array(max_drawdowns))
    ) / 6
    
    # Ensure all weights are positive
    composite_scores = np.maximum(composite_scores, 0)
    weights = composite_scores / composite_scores.sum()
    
    # Implement quarterly rebalancing
    rebalance_dates = pd.date_range(start=returns.index[0], end=returns.index[-1], freq='Q')
    portfolio_values = []
    current_weights = weights
    current_value = 10000 # Initial capital
    
    # Transaction cost (0.1% per trade)
    transaction_cost = 0.001
    
    # Add tracking for weight history
    weight_history = {stock: [] for stock in stocks}
    weight_dates = []
    
    print("Backtesting portfolio with quarterly rebalancing...")
    # Backtest with quarterly rebalancing
    for i in range(len(returns.index)):
        date = returns.index[i]
        
        # Calculate daily returns for the portfolio
        daily_return = np.sum(returns.iloc[i].values * current_weights)
        
        # Update portfolio value
        current_value *= (1 + daily_return)
        portfolio_values.append(current_value)
        
        # Check if it's time to rebalance (quarterly)
        if date in rebalance_dates:
            print(f"Rebalancing portfolio on {date.strftime('%Y-%m-%d')}...")
            
            # Calculate new weights based on data up to this point
            historical_returns = returns.loc[:date]
            
            # Recalculate parameters
            sharpe_ratios_rebal = (historical_returns.mean() - risk_free_rate) / historical_returns.std() * np.sqrt(252)
            
            # Recalculate betas
            betas_rebal = []
            for stock in stocks:
                common_idx = historical_returns.index.intersection(market_returns.loc[:date].index)
                x = market_returns.loc[common_idx].values.reshape(-1)
                y = historical_returns[stock].loc[common_idx].values.reshape(-1)
                beta = np.polyfit(x, y, 1)[0]
                betas_rebal.append(beta)
            
            # Recalculate drawdowns
            max_drawdowns_rebal = []
            for stock in stocks:
                cumulative_returns = (1 + historical_returns[stock]).cumprod()
                running_max = cumulative_returns.cummax()
                drawdown = (cumulative_returns / running_max) - 1
                max_drawdowns_rebal.append(abs(drawdown.min()))
            
            # Get updated sentiment scores
            all_news_rebal = get_news_from_yahoo_finance(stocks)
            sentiment_scores_dict_rebal = analyze_sentiment(all_news_rebal)
            sentiment_scores_rebal = np.array([sentiment_scores_dict_rebal[stock] for stock in stocks])
            
            # Get updated ESG scores
            esg_scores_dict_rebal = predict_esg_scores(all_news_rebal)
            esg_scores_rebal = np.array([esg_scores_dict_rebal[stock] for stock in stocks])
            
            # Recalculate volatility
            volatility_rebal = historical_returns.std().values
            
            # Calculate new composite scores
            composite_scores_rebal = (
                sentiment_scores_rebal +
                (1 / volatility_rebal) +
                esg_scores_rebal +
                sharpe_ratios_rebal.values +
                (1 / np.array(betas_rebal)) +
                (1 / np.array(max_drawdowns_rebal))
            ) / 6
            
            # Calculate new weights
            composite_scores_rebal = np.maximum(composite_scores_rebal, 0)
            new_weights = composite_scores_rebal / composite_scores_rebal.sum()
            
            # Calculate transaction costs
            total_trade_value = np.sum(np.abs(new_weights - current_weights)) * current_value
            transaction_cost_amount = total_trade_value * transaction_cost
            
            # Apply transaction costs
            current_value -= transaction_cost_amount
            
            # Update weights
            current_weights = new_weights
            
            # Track weight history
            weight_dates.append(date)
            for i, stock in enumerate(stocks):
                weight_history[stock].append(current_weights[i])
    
    # Convert portfolio values to a pandas Series
    portfolio_values = pd.Series(portfolio_values, index=returns.index)
    
    # Calculate portfolio metrics
    portfolio_returns = portfolio_values.pct_change().dropna()
    portfolio_sharpe = (portfolio_returns.mean() - risk_free_rate) / portfolio_returns.std() * np.sqrt(252)
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    portfolio_drawdown = (portfolio_cumulative / portfolio_cumulative.cummax()) - 1
    portfolio_max_drawdown = abs(portfolio_drawdown.min())
    
    # Plot the backtested performance
    plt.figure(figsize=(12, 8))
    plt.plot(portfolio_values, label='Enhanced SVE Strategy')
    plt.title('Backtested Enhanced SVE Strategy Performance')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid()
    plt.show()
    
    # Plot the weights over time
    plt.figure(figsize=(14, 8))
    for stock in stocks:
        plt.plot(weight_dates, weight_history[stock], marker='o', label=stock)
    
    plt.title('Portfolio Weights Over Time')
    plt.xlabel('Date')
    plt.ylabel('Weight')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Output weights and performance metrics
    print("\nFinal Stock Weights:")
    for stock, weight in zip(stocks, current_weights):
        print(f"{stock}: {weight:.4f}")
    
    print(f"\nPortfolio Metrics:")
    print(f"Initial Investment: $10,000.00")
    print(f"Final Portfolio Value: ${portfolio_values.iloc[-1]:.2f}")
    print(f"Portfolio Sharpe Ratio: {portfolio_sharpe:.4f}")
    print(f"Maximum Drawdown: {portfolio_max_drawdown:.4f} or {portfolio_max_drawdown*100:.2f}%")
    print(f"Total Return: {(portfolio_values.iloc[-1]/10000 - 1)*100:.2f}%")
    
    # Print sentiment and ESG scores
    print("\nSentiment Scores:")
    for stock, score in zip(stocks, sentiment_scores):
        print(f"{stock}: {score:.4f}")
    
    print("\nESG Scores:")
    for stock, score in zip(stocks, esg_scores):
        print(f"{stock}: {score:.4f}")
    
    # Print sample news used for sentiment analysis
    print("\nSample News Used for Sentiment Analysis:")
    for stock in stocks:
        if all_news[stock]:
            print(f"{stock}: {all_news[stock][0][:100]}...")
        else:
            print(f"{stock}: No news found")

if __name__ == "__main__":
    optimize_portfolio()
