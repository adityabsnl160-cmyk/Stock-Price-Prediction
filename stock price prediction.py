import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression

# Stock Data Download
data = yf.download("AAPL", start="2024-01-01", end="2025-01-01")

# Agar MultiIndex columns hain to unhe flatten karo
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Previous Day Close Price
data["Prev_Close"] = data["Close"].shift(1)

# Null values remove
data.dropna(inplace=True)

# Features and Target
X = data[["Prev_Close"]]
y = data["Close"]

# Model Training
model = LinearRegression()
model.fit(X, y)

# Next Day Prediction
last_price = float(data["Close"].iloc[-1])

# DataFrame ke form me input dena
next_day_input = pd.DataFrame(
    {"Prev_Close": [last_price]}
)

prediction = model.predict(next_day_input)

# Output
print("Last Closing Price:", round(last_price, 2))
print("Predicted Next Day Price:", round(prediction[0], 2))