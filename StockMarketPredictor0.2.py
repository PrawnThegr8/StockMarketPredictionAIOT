import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import streamlit as st
from datetime import date
import yfinance as yf
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import pandas as pd
import requests
from textblob import TextBlob

TODAY = date.today().strftime("%Y-%m-%d")
NEWS_API_KEY = 'd924dd3c445d430ba37bd28e3cd69e32'  # Replace with your News API key

st.title('Stock Market Predictor')

# Get the list of all available tickers and their corresponding long names
all_tickers_dict = yf.Tickers(list(yf.Tickers().tickers.keys())).tickers
all_tickers = list(all_tickers_dict.keys())

selected_stock = st.selectbox('Select a stock ticker for prediction', all_tickers)

start_year = st.slider('Select the start year for prediction', 2010, date.today().year - 1, 2020)

start_date = date(start_year, 1, 1).strftime("%Y-%m-%d")

n_years = st.slider('How many years into the future?', 1, 4)
period = n_years * 365

daily_data = None  # Initialize daily_data here

def load_data(ticker):
    if selected_stock:
        data = yf.download(ticker, start_date, TODAY)
        data.reset_index(inplace=True)
        return data, all_tickers_dict

@st.cache_data
def get_news(stock):
    if NEWS_API_KEY:
        # Include the company name along with the stock ticker in the search query
        company_name = yf.Ticker(stock).info['longName']
        search_query = f'{stock} OR {company_name}'
        
        url = f'https://newsapi.org/v2/everything?q={search_query}&apiKey={NEWS_API_KEY}&pageSize=5'
        response = requests.get(url)
        news_data = response.json()
        return news_data

# Custom word lists for positive and negative sentiment
positive_words = ['good', 'excellent', 'positive', 'improve', 'success', 'up', 'gain', 'bullish', 'happy', 'prosper', 'opportunity']
negative_words = ['bad', 'poor', 'negative', 'decline', 'failure', 'down', 'loss', 'bearish', 'sad', 'danger', 'risk']

def analyze_sentiment(text):
    blob = TextBlob(text.lower())
    sentiment_score = 0

    # Check each word in the text and update the sentiment score based on custom word lists
    for word in blob.words:
        if word in positive_words:
            sentiment_score += 1
        elif word in negative_words:
            sentiment_score -= 1

    return sentiment_score

if selected_stock:
    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    smoothing_factor = st.slider('Smoothing Factor (increase for smoother graph)', 0.1, 0.95, 0.9, 0.05)
    changepoint_prior_scale = st.slider('Flexibility of Trend', 0.1, 10.0, 0.5, 0.1, format="%.1f")

    # Daily data for exponential smoothing
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    daily_data = data.resample('D').interpolate()
    daily_data['Close_rolling'] = daily_data['Close'].ewm(alpha=1 - smoothing_factor).mean()

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Open'], name="Stock Open"))
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Close'], name="Stock Close"))
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Close_rolling'], name="Close (Exponential Smoothing)"))
        fig.update_layout(
            title_text='Stock History',
            xaxis_rangeslider_visible=True,
            height=600,
            width=900
        )
        st.plotly_chart(fig)

    plot_raw_data()

    news_data = get_news(selected_stock)

    overall_sentiment_score = 0
    if 'articles' in news_data and len(news_data['articles']) > 0:
        for article in news_data['articles']:
            # Check if the description has a minimum word count to consider it relevant
            min_word_count = 10  # Experiment with different values here
            if len(article['description'].split()) >= min_word_count:
                st.write(f"**Title:** {article['title']}")
                st.write(f"**Description:** {article['description']}")
                st.write(f"**Source:** {article['source']['name']}")
                st.write(f"**Published At:** {article['publishedAt']}")
                st.write(f"**URL:** {article['url']}")

                sentiment_score = analyze_sentiment(article['description'])

                # Adjust sensitivity by multiplying with a weight
                weight = 10  # Experiment with different values here for sensitivity
                sentiment_score *= weight

                overall_sentiment_score += sentiment_score

    if overall_sentiment_score > 0:
        st.subheader("Overall Sentiment: Positive")
    elif overall_sentiment_score < 0:
        st.subheader("Overall Sentiment: Negative")
    else:
        st.subheader("Overall Sentiment: Neutral")

    df_train = daily_data[['Close_rolling']].reset_index().rename(columns={"Date": "ds", "Close_rolling": "y"})

    m = Prophet(
        growth='linear',
        changepoint_prior_scale=changepoint_prior_scale
    )

    m.fit(df_train)

    future = m.make_future_dataframe(periods=period, freq='D')

    forecast = m.predict(future)

    if n_years == 1:
        st.subheader(f'Forecast Plot for {n_years} Year')
    else:
        st.subheader(f'Forecast Plot for {n_years} Years')

    fig1 = plot_plotly(m, forecast)

    fig1.update_traces(mode='lines', line=dict(color='blue', width=2), selector=dict(name='yhat'))

    num_data_points = len(forecast)
    marker_size = max(4, 200 // num_data_points)

    fig1.update_traces(mode='markers+lines', marker=dict(size=marker_size, color='black', opacity=0.7),
                       selector=dict(name='yhat_lower,yhat_upper'))

    fig1.update_layout(
        title_text=f'Forecast Plot for {n_years} Years',
        xaxis_rangeslider_visible=True,
        height=600,
        width=900,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig1)

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
