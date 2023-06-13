# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 00:21:12 2023

@author: ajeeth
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

# Load and preprocess data
data = pd.read_csv('crude-oil-price.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
time_series = data['price']

# Set the frequency of the time series (assuming daily frequency)
time_series.index = pd.date_range(start=time_series.index[0], periods=len(time_series), freq='D')

# Apply seasonal decomposition of time series
result = seasonal_decompose(time_series, model='additive')

# Retrieve the components from decomposition
trend = result.trend
seasonal = result.seasonal
residual = result.resid

# Forecasting
model = ARIMA(time_series, order=(1, 0, 1))
model_fit = model.fit()

future_dates_15 = pd.date_range(start=time_series.index[-1], periods=15, freq='D')
forecast_15 = model_fit.predict(start=len(time_series), end=len(time_series) + 14)

future_dates_30 = pd.date_range(start=time_series.index[-1], periods=30, freq='D')
forecast_30 = model_fit.predict(start=len(time_series), end=len(time_series) + 29)

# Streamlit app
def main():
    st.title('Crude Oil Price Analysis')

    st.subheader('Original Time Series')
    st.line_chart(time_series)

    st.subheader('Trend Component')
    st.line_chart(trend)

    st.subheader('Seasonal Component')
    st.line_chart(seasonal)

    st.subheader('Residual Component')
    st.line_chart(residual)

    st.subheader('Forecast for Next 15 Days')
    forecast_df_15 = pd.DataFrame({'Date': future_dates_15, 'Forecast': forecast_15})
    st.line_chart(forecast_df_15.set_index('Date'))

    st.subheader('Forecast for Next 30 Days')
    forecast_df_30 = pd.DataFrame({'Date': future_dates_30, 'Forecast': forecast_30})
    st.line_chart(forecast_df_30.set_index('Date'))

    st.subheader('Predict Oil Price for a Specific Date')
    selected_date = st.date_input('Select a date')
    if selected_date:
        selected_date = pd.Timestamp(selected_date)  # Convert selected_date to Timestamp
        prediction = model_fit.predict(start=len(time_series), end=len(time_series) + (selected_date - time_series.index[-1]).days)
        st.write('The predicted oil price for the selected date is:', prediction[-1])

if __name__ == '__main__':
    main()
