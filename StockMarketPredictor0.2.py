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

    # Plot raw data with exponential smoothing
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Open'], name="Stock Open"))
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Close'], name="Stock Close"))
        fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Close_rolling'], name="Close (Exponential Smoothing)"))
        fig.update_layout(
            title_text='Stock History',
            xaxis_rangeslider_visible=True,
            height=600,  # Set the desired height for the raw data plot
            width=900  # Set the desired width for the raw data plot
        )
        st.plotly_chart(fig)

    plot_raw_data()

    # Prepare additional features (e.g., daily returns, moving averages, etc.)
    daily_data['Daily_Return'] = daily_data['Close'].pct_change()
    daily_data['MA_50'] = daily_data['Close'].rolling(window=50).mean()
    daily_data['MA_200'] = daily_data['Close'].rolling(window=200).mean()

    # Predict forecast with Prophet
    df_train = daily_data[['Close_rolling', 'Daily_Return', 'MA_50', 'MA_200']].reset_index().rename(
        columns={"Date": "ds", "Close_rolling": "y", "Daily_Return": "extra_regressor1", "MA_50": "extra_regressor2",
                 "MA_200": "extra_regressor3"})

    m = Prophet(
        growth='linear',
        changepoint_prior_scale=changepoint_prior_scale
    )

    m.add_regressor('extra_regressor1')
    m.add_regressor('extra_regressor2')
    m.add_regressor('extra_regressor3')

    m.fit(df_train)

    future = m.make_future_dataframe(periods=period, freq='D')

    forecast = m.predict(future)

    # LSTM Model for Comparison
    def create_sequences(data, seq_length):
        X = []
        y = []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    sequence_length = 30
    X, y = create_sequences(daily_data['Close'].values, sequence_length)

    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential()
    model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # ... (Rest of the LSTM predictions and transformation back to original scale)

    # Plot the LSTM predictions
    fig_lstm = go.Figure()
    fig_lstm.add_trace(go.Scatter(x=train_dates, y=y_train.flatten(), name='Train Actual', line=dict(color='blue')))
    fig_lstm.add_trace(
        go.Scatter(x=train_dates, y=train_predictions.flatten(), name='Train Predicted', line=dict(color='orange')))
    fig_lstm.add_trace(go.Scatter(x=test_dates, y=y_test.flatten(), name='Test Actual', line=dict(color='green')))
    fig_lstm.add_trace(go.Scatter(x=test_dates, y=test_predictions.flatten(), name='Test Predicted', line=dict(color='red')))
    fig_lstm.update_layout(title_text='Stock Price Prediction with LSTM', xaxis_title='Date', yaxis_title='Stock Price')
    st.plotly_chart(fig_lstm)

    # ... (Rest of the code remains the same)

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
