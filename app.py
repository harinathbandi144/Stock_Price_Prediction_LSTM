import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle

st.title("📈 Stock Price Prediction App")

# Sidebar inputs
stock = st.sidebar.text_input("Enter Stock Ticker", "POWERGRID.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2000-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-11-01"))

# Load data
df = yf.download(stock, start=start_date, end=end_date)
st.subheader("Raw Data")
st.write(df.tail())

# Load pre-trained model
with open("stock_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare test data
train_size = int(len(df)*0.70)
training_data = pd.DataFrame(df['Close'][0:train_size])
testing_data = pd.DataFrame(df['Close'][train_size:])

scaler = MinMaxScaler()
past_data_100 = training_data.tail(100)
final_df = pd.concat([past_data_100, testing_data], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Predict
y_pred = model.predict(x_test)
scale_factor = 1/scaler.scale_[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

# Plot predictions
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(y_test, label="Original Price")
ax.plot(y_pred, label="Predicted Price")
ax.legend()
st.pyplot(fig)
