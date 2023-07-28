import streamlit as st
from datetime import date
import yfinance as yf
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import pandas as pd
import requests
from transformers import pipeline

TODAY = date.today().strftime("%Y-%m-%d")
NEWS_API_KEY = 'd924dd3c445d430ba37bd28e3cd69e32'  # Replace with your News API key

st.title('Stock Market Predictor')

# Use selectbox for stock selection
selected_stock = st.text_input('Select a stock ticker for prediction (refer to yfinance for ticker)')

start_year = st.slider('Select the start year for prediction', 2010, date.today().year - 1, 2020)

start_date = date(start_year, 1, 1).strftime("%Y-%m-%d")

n_years = st.slider('How many years into the future?', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    if selected_stock:
        data = yf.download(ticker, start_date, TODAY)
        data.reset_index(inplace=True)
        return data

@st.cache_data
def get_news(stock):
    if NEWS_API_KEY:
        url = f'https://newsapi.org/v2/everything?q={stock}&apiKey={NEWS_API_KEY}&pageSize=5'
        response = requests.get(url)
        news_data = response.json()
        return news_data

def get_sentiment_analysis(text):
    sentiment_analyzer = pipeline(task='sentiment-analysis', model='distilbert-base-uncased')
    result = sentiment_analyzer(text)
    return result[0]['label']

if selected_stock:
    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    # Apply exponential smoothing to the data
    smoothing_factor = st.slider('Smoothing Factor (increase for smoother graph)', 0.1, 0.95, 0.9, 0.05)
    changepoint_prior_scale = st.slider('Flexibility of Trend', 0.1, 10.0, 0.5, 0.1, format="%.1f")

    # ... (Remaining code remains the same)

    # Fetch news for the selected stock
    news_data = get_news(selected_stock)

    # Analyze news sentiment and determine overall sentiment
    overall_sentiment = 0
    if 'articles' in news_data and len(news_data['articles']) > 0:
        for article in news_data['articles']:
            st.write(f"**Title:** {article['title']}")
            st.write(f"**Description:** {article['description']}")
            st.write(f"**Source:** {article['source']['name']}")
            st.write(f"**Published At:** {article['publishedAt']}")
            st.write(f"**URL:** {article['url']}")
            st.write('---')

            # Analyze sentiment of the article's description
            sentiment = get_sentiment_analysis(article['description'])
            if sentiment == 'POSITIVE':
                overall_sentiment += 1
            elif sentiment == 'NEGATIVE':
                overall_sentiment -= 1

    # Determine overall sentiment
    if overall_sentiment > 0:
        st.subheader("Overall Sentiment: Positive")
    elif overall_sentiment < 0:
        st.subheader("Overall Sentiment: Negative")
    else:
        st.subheader("Overall Sentiment: Neutral")

    # ... (Remaining code remains the same)

    # Show and plot forecast
    if n_years == 1:
        st.subheader(f'Forecast Plot for {n_years} Year')
    else:
        st.subheader(f'Forecast Plot for {n_years} Years')

    fig1 = plot_plotly(m, forecast)

    # ... (Remaining code remains the same)

    st.plotly_chart(fig1)

# Footer
footer = """
<style>
.footer {
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
}
</style>
<div class="footer">
    <p>Coded by Pranav, Ideas by Emil</p>
    <p>This app is made for educational purposes only. Data it provides is not 100% accurate.</p>
    <p>Analyze stocks before investing.</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
