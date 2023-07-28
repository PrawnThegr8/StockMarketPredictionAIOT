import streamlit as st
from datetime import date
import yfinance as yf
import numpy as np
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
    
    # Convert 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Set 'Date' column as index
    data.set_index('Date', inplace=True)

    # Resample the data to daily frequency and fill missing values with the previous day's price
    data_daily = data.resample('D').ffill()

    # Use 'Close' price for prediction
    prices = data_daily['Close'].values.reshape(-1, 1)

    # Normalize the data manually
    max_price = data_daily['Close'].max()
    min_price = data_daily['Close'].min()
    scaled_prices = (prices - min_price) / (max_price - min_price)

    # Create sequences for LSTM training
    def create_sequences(data, seq_length):
        X = []
        y = []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    sequence_length = 30  # Experiment with different sequence lengths
    X, y = create_sequences(scaled_prices, sequence_length)

    # Split data into training and testing sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Transform the predictions back to the original scale
    train_predictions = train_predictions * (max_price - min_price) + min_price
    y_train = y_train * (max_price - min_price) + min_price
    test_predictions = test_predictions * (max_price - min_price) + min_price
    y_test = y_test * (max_price - min_price) + min_price

    # Create a DataFrame for plotting
    train_dates = data_daily.index[sequence_length:train_size + sequence_length]
    test_dates = data_daily.index[train_size + sequence_length:]
    train_data = pd.DataFrame({'Date': train_dates, 'Actual': y_train.flatten(), 'Predicted': train_predictions.flatten()})
    test_data = pd.DataFrame({'Date': test_dates, 'Actual': y_test.flatten(), 'Predicted': test_predictions.flatten()})

    # Plot the predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_data['Date'], y=train_data['Actual'], name='Train Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=train_data['Date'], y=train_data['Predicted'], name='Train Predicted', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Actual'], name='Test Actual', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Predicted'], name='Test Predicted', line=dict(color='red')))
    fig.update_layout(title_text='Stock Price Prediction with LSTM', xaxis_title='Date', yaxis_title='Stock Price')
    st.plotly_chart(fig)
