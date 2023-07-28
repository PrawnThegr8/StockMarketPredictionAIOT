import streamlit as st
from datetime import date
import yfinance as yf
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objs as go
import pandas as pd

TODAY = date.today().strftime("%Y-%m-%d")

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

if selected_stock:
    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')
    
    # Apply exponential smoothing to the data
    smoothing_factor = st.slider('Smoothing Factor (increase for smoother graph)', 0.1, 0.95, 0.9, 0.05)
    changepoint_prior_scale = st.slider('Flexibility of Trend', 0.1, 10.0, 0.5, 0.1, format="%.1f")

    # Convert 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Set 'Date' column as index
    data.set_index('Date', inplace=True)

    # Resample the data to daily frequency
    daily_data = data.resample('D').interpolate()

    # Apply exponential smoothing to the data
    daily_data['Close_rolling'] = daily_data['Close'].ewm(alpha=1 - smoothing_factor).mean()

    # Prepare additional features (e.g., daily returns, moving averages, etc.)
    daily_data['Daily_Return'] = daily_data['Close'].pct_change()

    # Fill missing values in 'Daily_Return' using interpolation
    daily_data['Daily_Return'].interpolate(method='linear', inplace=True)

    # Create 'extra_regressor1' column and fill missing values with a specific value (e.g., 0)
    daily_data['extra_regressor1'] = daily_data['Daily_Return'].fillna(0)

    daily_data['MA_50'] = daily_data['Close'].rolling(window=50).mean()
    daily_data['MA_200'] = daily_data['Close'].rolling(window=200).mean()

    # Plot raw data with exponential smoothing and additional features
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Open'], name="Stock Open"))
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Close'], name="Stock Close"))
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Close_rolling'], name="Close (Exponential Smoothing)"))
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['MA_50'], name="MA_50"))
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['MA_200'], name="MA_200"))
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['extra_regressor1'], name="Extra Regressor 1"))
        fig.update_layout(
            title_text='Stock History',
            xaxis_rangeslider_visible=True,
            height=600,  # Set the desired height for the raw data plot
            width=900  # Set the desired width for the raw data plot
        )
        st.plotly_chart(fig)

    plot_raw_data()

    # Predict forecast with Prophet
    df_train = daily_data[['Close_rolling']].reset_index().rename(columns={"Date": "ds", "Close_rolling": "y"})

    m = Prophet(
        growth='linear',
        changepoint_prior_scale=changepoint_prior_scale
    )

    # Add additional regressors if available
    m.add_regressor('extra_regressor1')
    # You can add more additional regressors here if needed

    m.fit(df_train)

    future = m.make_future_dataframe(periods=period, freq='D')

    # Fine-tune seasonality parameters if needed
    # m.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.1)
    # m.add_seasonality(name='yearly', period=365.25, fourier_order=10, prior_scale=0.1)

    forecast = m.predict(future)

    # Show and plot forecast
    if n_years == 1:
        st.subheader(f'Forecast Plot for {n_years} Year')
    else:
        st.subheader(f'Forecast Plot for {n_years} Years')

    fig1 = plot_plotly(m, forecast)

    # Customize the forecast line appearance
    fig1.update_traces(mode='lines', line=dict(color='blue', width=2), selector=dict(name='yhat'))

    # Calculate the marker size based on the number of data points
    num_data_points = len(forecast)
    marker_size = max(4, 200 // num_data_points)  # Adjust the factor as needed

    # Update the scatter trace to match the forecast more closely
    fig1.update_traces(mode='markers+lines', marker=dict(size=marker_size, color='black', opacity=0.7),
                       selector=dict(name='yhat_lower,yhat_upper'))

    fig1.update_layout(
        title_text=f'Forecast Plot for {n_years} Years',
        xaxis_rangeslider_visible=True,
        height=600,  # Set the desired height for the forecast plot
        width=900,  # Set the desired width for the forecast plot
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

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
    <p>This app is made for educational purposes only. Data it provides is not 
