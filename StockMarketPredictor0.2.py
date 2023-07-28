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
        # Include the company name along with the stock ticker in the search query
        company_name = yf.Ticker(stock).info['longName']
        search_query = f'{stock} OR {company_name}'
        
        url = f'https://newsapi.org/v2/everything?q={search_query}&apiKey={NEWS_API_KEY}&pageSize=5'
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

    smoothing_factor = st.slider('Smoothing Factor (increase for smoother graph)', 0.1, 0.95, 0.9, 0.05)
    changepoint_prior_scale = st.slider('Flexibility of Trend', 0.1, 10.0, 0.5, 0.1, format="%.1f")

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

    overall_sentiment = 0
    if 'articles' in news_data and len(news_data['articles']) > 0:
        for article in news_data['articles']:
            st.write(f"**Title:** {article['title']}")
            st.write(f"**Description:** {article['description']}")
            st.write(f"**Source:** {article['source']['name']}")
            st.write(f"**Published At:** {article['publishedAt']}")
            st.write(f"**URL:** {article['url']}")
            st.write('---')

            sentiment = get_sentiment_analysis(article['description'])
            if sentiment == 'POSITIVE':
                overall_sentiment += 1
            elif sentiment == 'NEGATIVE':
                overall_sentiment -= 1

    if overall_sentiment > 0:
        st.subheader("Overall Sentiment: Positive")
    elif overall_sentiment < 0:
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
