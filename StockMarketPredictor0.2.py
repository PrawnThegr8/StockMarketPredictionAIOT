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
    
    # ... (Rest of the code remains the same)

    # Plot the predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_data['Date'], y=train_data['Actual'], name='Train Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=train_data['Date'], y=train_data['
