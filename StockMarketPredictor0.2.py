import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import pandas as pd

START = "2021-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Market Predictor')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'AMZN', 'RELIANCE.NS')
selected_stock = st.text_input('Select a stock ticker for prediction (refer YAHOO FINANCE for tickers)')

n_years = st.slider('How many years into the future?', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    if selected_stock:
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

if selected_stock:
    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    # Apply exponential smoothing to the data
    smoothing_factor = st.slider('Smoothing Factor (increase for smoother graph)', 0.1, 0.95, 0.9, 0.05)

# Convert 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

# Set 'Date' column as index
    data.set_index('Date', inplace=True)

# Resample the data to daily frequency
    daily_data = data.resample('D').interpolate()

# Apply exponential smoothing to the data
    daily_data['Close_rolling'] = daily_data['Close'].ewm(alpha=1 - smoothing_factor).mean()

# Plot raw data with exponential smoothing
def plot_raw_data():
    if selected_stock:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Open'], name="Stock Open"))
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Close'], name="Stock Close"))
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Close_rolling'], name="Close (Exponential Smoothing)"))
        fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
if selected_stock:
    plot_raw_data()

# Predict forecast with Prophet
    df_train = daily_data[['Close_rolling']].reset_index().rename(columns={"Date": "ds", "Close_rolling": "y"})

    m = Prophet(changepoint_prior_scale=0.90)  # Adjust the value as needed
    m.fit(df_train)

    future = m.make_future_dataframe(periods=period, freq='D')
    forecast = m.predict(future)

# Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.subheader(f'Forecast Plot for {n_years} Years')
    fig1 = plot_plotly(m, forecast)
    fig1.update_layout(yaxis=dict(tickformat=".2f"))
    st.plotly_chart(fig1)

    st.subheader('Forecast Components')
    fig2 = m.plot_components(forecast)
    st.write(fig2)

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
