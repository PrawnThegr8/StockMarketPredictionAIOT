# pip install streamlit fbprophet yfinance plotly
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
selected_stock = st.text_input('Select a stock ticker for prediction(refer YAHOO FINANCE for tickers)')

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
    
    st.subheader('Raw data')
    st.write(data.tail())

# Apply rolling average and interpolation to the data
rolling_window = st.slider('Rolling average of data(increase for smoother graph)', 1, 30)
if selected_stock:
    data['Close_rolling'] = data['Close'].rolling(rolling_window, min_periods=1).mean()
    data['Close_rolling'] = data['Close_rolling'].interpolate(method='linear')
    
# Plot raw data with rolling average
def plot_raw_data():
    if selected_stock:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close_rolling'], name=f"Close ({rolling_window}D Rolling Avg)"))
        fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
if selected_stock:
    plot_raw_data()

# Predict forecast with Prophet
if selected_stock:
    df_train = data[['Date', 'Close_rolling']].rename(columns={"Date": "ds", "Close_rolling": "y"})

    m = Prophet(changepoint_prior_scale=0.05)
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
    bottom:;
    width: 100%;

    color: white;
    text-align: center;
}
</style>
<div class="footer">
    <p>Coded by Pranav, Ideas by Emil</p>
    <p>This app is made for educational purposes only. Data it provides is not 100% accurate. </p>
    <p>Analyze stocks before investing.</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
