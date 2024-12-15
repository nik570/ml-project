import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime

# Step 1: Fetch stock data
try:
    ticker = input("Enter the stock ticker symbol (e.g., TATASTEEL.NS): ").strip().upper()
    print("\nFetching data for", ticker, "from Yahoo Finance...")

    # Get today's date to use as the 'end' parameter
    today = datetime.today().strftime('%Y-%m-%d')

    # Download stock data up to today
    data = yf.download(ticker, start='2015-01-01', end=today)

    if data.empty:
        raise ValueError("No data found for the given ticker. Please check the ticker symbol and try again.")

    # Prepare data
    data['Date'] = data.index
    data.reset_index(drop=True, inplace=True)

    # Ask the user if they want to view the data
    view_data = input("Do you want to view the fetched data? (yes/no): ").strip().lower()
    if view_data in ['yes', 'y']:
        print("\n--- Historical Stock Data ---")
        print(data.tail(10))  # Display the last 10 rows of the data
except Exception as e:
    print("Error occurred while fetching data:", str(e))
    exit()

# Step 2: Feature Engineering - Use past 5 days to predict the next day
data['Target'] = data['Close'].shift(-1)  # Target is the next day's close price
for i in range(1, 6):
    data[f'Lag_{i}'] = data['Close'].shift(i)

data.dropna(inplace=True)  # Drop rows with NaN values

# Step 3: Split data into training and testing sets
X = data[['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5']]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Neural Network (MLP) model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Single output for stock price prediction
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Step 5: Make predictions for the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"\nModel Evaluation - Mean Squared Error: {mse:.2f}")

# Step 6: Predict tomorrow's price using the latest data
try:
    latest_data = X.iloc[-1].values.reshape(1, -1)  # Preparing the latest data for prediction
    tomorrow_predicted = model.predict(latest_data)[0][0]  # Extract single prediction value
    today_close = float(data['Close'].iloc[-1])  # Convert to scalar (float)
except Exception as e:
    print("Error during prediction:", str(e))
    exit()

# Text Output: Prediction Summary
print("\n--- Prediction Summary ---")
print(f"Today's Closing Price: {today_close:.2f}")
print(f"Tomorrow's Predicted Closing Price: {tomorrow_predicted:.2f}")

if tomorrow_predicted > today_close:
    print("Prediction: The stock price is expected to INCREASE tomorrow.")
else:
    print("Prediction: The stock price is expected to DECREASE tomorrow.")

# Step 7: Visualize predictions vs actual values
# Flatten y_test and y_pred for visualization
y_test_flat = y_test.values.flatten()[:100]  # Only taking the first 100 values
y_pred_flat = y_pred.flatten()[:100]  # Only taking the first 100 values

# Ensure lengths match for plotting
min_length = min(len(y_test_flat), len(y_pred_flat))
y_test_flat = y_test_flat[:min_length]
y_pred_flat = y_pred_flat[:min_length]

# Plot
plt.figure(figsize=(12, 6))  # Wider figure for better readability

# Plot actual values (y_test)
plt.plot(y_test_flat, label='Actual Prices', color='blue', linestyle='-', linewidth=2, marker='o', markersize=4, alpha=0.8)

# Plot predicted values (y_pred)
plt.plot(y_pred_flat, label='Predicted Prices', color='red', linestyle='--', linewidth=2, marker='x', markersize=6, alpha=0.8)

# Highlight the difference using a fill
plt.fill_between(range(len(y_test_flat)), y_test_flat, y_pred_flat, color='gray', alpha=0.3, label='Error Margin')

# Add titles, labels, and grid
plt.title(f'{ticker} Stock Price Prediction (Neural Network) - First {min_length} Samples', fontsize=16)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Stock Price (INR)', fontsize=12)

# Add legend
plt.legend(loc='best', fontsize=10)

# Customize grid
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
