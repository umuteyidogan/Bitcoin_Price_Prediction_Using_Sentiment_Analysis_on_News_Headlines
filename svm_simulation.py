import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ta
import numpy as np
import matplotlib.pyplot as plt

# Load the sentiment data from the Excel file
sentiment_file_path = '/Users/umuteyidogan/Desktop/IGP_Project/Daily_Sentiment_Analysis_Lem_Headline.xlsx'
sentiment_data = pd.read_excel(sentiment_file_path)

# Load the Bitcoin price data from the CSV file
bitcoin_file_path = '/Users/umuteyidogan/Desktop/IGP_Project/bitcoin_price_with_5_labels_2.csv'
bitcoin_data = pd.read_csv(bitcoin_file_path)

# Load the trading volume data with labels from the CSV file
trading_volume_file_path = '/Users/umuteyidogan/Desktop/IGP_Project/trading_volume_with_labels.csv'
trading_volume_data = pd.read_csv(trading_volume_file_path)

# Ensure the date formats are consistent and convert to datetime
sentiment_data['Published date'] = pd.to_datetime(sentiment_data['Published date'])
bitcoin_data['Date'] = pd.to_datetime(bitcoin_data['Date'])
trading_volume_data['Date'] = pd.to_datetime(trading_volume_data['Date'])

# Merge the sentiment data with the Bitcoin price data on the date
merged_data = pd.merge(sentiment_data, bitcoin_data, left_on='Published date', right_on='Date', how='inner')

# Merge the resulting data with the trading volume data
final_data = pd.merge(merged_data, trading_volume_data, on='Date', how='inner')

# Calculate technical indicators
final_data['SMA_7'] = ta.trend.sma_indicator(final_data['Close'], window=7)
final_data['EMA_14'] = ta.trend.ema_indicator(final_data['Close'], window=14)
final_data['RSI'] = ta.momentum.rsi(final_data['Close'], window=14)
final_data['MACD'] = ta.trend.macd(final_data['Close'])
final_data['MACD_Signal'] = ta.trend.macd_signal(final_data['Close'])
final_data['Bollinger_High'] = ta.volatility.bollinger_hband(final_data['Close'])
final_data['Bollinger_Low'] = ta.volatility.bollinger_lband(final_data['Close'])

# Drop rows with NaN values caused by the indicators calculation
final_data = final_data.dropna()

# Select relevant columns for the final dataset
final_data = final_data[['Published date', 'Positive_Percentage', 'Negative_Percentage', 'Neutral_Percentage', 
                         'Volume', 'SMA_7', 'EMA_14', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_High', 
                         'Bollinger_Low', 'Close', 'Label']]

# Shuffle the dataset to remove any ordering bias
final_data = final_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Define the features and target variable
features = final_data[['Positive_Percentage', 'Negative_Percentage', 'Neutral_Percentage', 'Volume', 
                       'SMA_7', 'EMA_14', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_High', 
                       'Bollinger_Low']]
target = final_data['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, close_train, close_test = train_test_split(features, target, final_data['Close'], test_size=0.2, random_state=42)

# Define a pipeline with a scaler and SVM classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('svc', SVC(probability=True))  # SVM Classifier with probability estimates
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'svc__C': [0.1, 1, 10, 100],  # Regularization parameter
    'svc__gamma': [1, 0.1, 0.01, 0.001],  # Kernel coefficient
    'svc__kernel': ['linear', 'rbf']  # Specifies the kernel type to be used in the algorithm
}

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train the SVM model with the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Predict the probabilities on the test set
y_pred_prob = best_model.predict_proba(X_test)

# Convert predicted probabilities to integer labels (predictions)
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display the results
print(f"Best parameters: {best_params}")
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Verify that the confusion matrix sums up to the total number of test instances
print(f"Total number of test instances: {len(y_test)}")
print(f"Sum of confusion matrix values: {conf_matrix.sum()}")

# Trading simulation
# Initialize parameters
initial_capital = 10000.0  # Starting capital for each person, ensure it is a float
num_people = 100  # Number of people in each group
trade_amount = 1000  # Amount to trade each time (Notional value of each trade)
num_trades = len(y_test)  # Number of trades is 390

# Ensure trades are integers
model_trades = np.tile(y_pred, (num_people, 1)).astype(int)

# Simulate random trading
np.random.seed(42)
random_trades = np.random.choice([0, 1, 2, 3, 4], size=(num_people, num_trades)).astype(int)  # Random decisions

# Function to simulate trades with new labels
def simulate_trades(trades, prices):
    capital = np.full(trades.shape[0], initial_capital, dtype=np.float64)
    for i in range(1, trades.shape[1]):  # Start from 1 to avoid index error
        # Calculate the percentage change in price
        pct_change = (prices[i] - prices[i - 1]) / prices[i - 1]

        for j in range(trades.shape[0]):
            if trades[j, i] == 0:  # Strong Sell
                capital[j] -= 2 * trade_amount * pct_change
            elif trades[j, i] == 1:  # Sell
                capital[j] -= trade_amount * pct_change
            elif trades[j, i] == 2:  # Hold
                continue
            elif trades[j, i] == 3:  # Buy
                capital[j] += trade_amount * pct_change
            elif trades[j, i] == 4:  # Strong Buy
                capital[j] += 2 * trade_amount * pct_change

    return capital

# Use the Close prices from the test set
prices = close_test.values

# Simulate random trades
random_capital_end = simulate_trades(random_trades, prices)

# Simulate model-based trades
model_capital_end = simulate_trades(model_trades, prices)

# Calculate average ending capital for both strategies
random_average_end_capital = np.mean(random_capital_end)
model_average_end_capital = np.mean(model_capital_end)

# Display the results
print(f"Average ending capital for random strategy: ${random_average_end_capital:.2f}")
print(f"Average ending capital for model-based strategy: ${model_average_end_capital:.2f}")

# Additional debugging info
print(f"Random strategy capital range: {random_capital_end.min()} to {random_capital_end.max()}")
print(f"Model-based strategy capital range: {model_capital_end.min()} to {model_capital_end.max()}")

# Plotting the results
plt.figure(figsize=(12, 6))

# Adjust the bins to capture the distributions better
bins = np.linspace(-100000, 200000, 100)

plt.hist(random_capital_end, bins=bins, alpha=0.7, label='Random Strategy')
plt.hist(model_capital_end, bins=bins, alpha=0.7, label='Model-Based Strategy')

plt.axvline(random_average_end_capital, color='blue', linestyle='dashed', linewidth=1)
plt.axvline(model_average_end_capital, color='orange', linestyle='dashed', linewidth=1)

plt.xlabel('Ending Capital')
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution of Ending Capital for Random and Model-Based Strategies')
plt.show()

# Let's assume simulate_trades is a function that returns capital over time instead of just the final capital
def simulate_trades_over_time(trades, prices):
    capital = np.full((trades.shape[0], trades.shape[1]), initial_capital, dtype=np.float64)
    for i in range(1, trades.shape[1]):
        pct_change = (prices[i] - prices[i - 1]) / prices[i - 1]
        for j in range(trades.shape[0]):
            if trades[j, i] == 0:  # Strong Sell
                capital[j, i] = capital[j, i - 1] - 2 * trade_amount * pct_change
            elif trades[j, i] == 1:  # Sell
                capital[j, i] = capital[j, i - 1] - trade_amount * pct_change
            elif trades[j, i] == 2:  # Hold
                capital[j, i] = capital[j, i - 1]
            elif trades[j, i] == 3:  # Buy
                capital[j, i] = capital[j, i - 1] + trade_amount * pct_change
            elif trades[j, i] == 4:  # Strong Buy
                capital[j, i] = capital[j, i - 1] + 2 * trade_amount * pct_change
    return capital

# Simulate trades over time for both strategies
random_capital_over_time = simulate_trades_over_time(random_trades, prices)
model_capital_over_time = simulate_trades_over_time(model_trades, prices)

# Calculate average capital over time
random_average_capital_over_time = np.mean(random_capital_over_time, axis=0)
model_average_capital_over_time = np.mean(model_capital_over_time, axis=0)

# Plot the average capital over time for both strategies
plt.figure(figsize=(12, 6))
plt.plot(random_average_capital_over_time, label='Random Strategy', color='red', linestyle='--')
plt.plot(model_average_capital_over_time, label='Model-based Strategy', color='blue')
plt.title('Trading Simulation: Capital Over Time')
plt.xlabel('Trade Number')
plt.ylabel('Average Capital ($)')
plt.legend()
plt.grid(True)
plt.show()
