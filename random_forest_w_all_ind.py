import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ta

# Load the sentiment data from the Excel file
sentiment_file_path = '/Users/umuteyidogan/Desktop/IGP_Project/Daily_Sentiment_Analysis_Lem_Headline.xlsx'
sentiment_data = pd.read_excel(sentiment_file_path)

# Load the Bitcoin price data from the CSV file
bitcoin_file_path = '/Users/umuteyidogan/Desktop/IGP_Project/bitcoin_price_with_labels_2.csv'
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
                         'Bollinger_Low', 'Label']]

# Shuffle the dataset to remove any ordering bias
final_data = final_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Define the features and target variable
features = final_data[['Positive_Percentage', 'Negative_Percentage', 'Neutral_Percentage', 'Volume', 
                       'SMA_7', 'EMA_14', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_High', 
                       'Bollinger_Low']]
target = final_data['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Define a pipeline with a scaler and random forest classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('rf', RandomForestClassifier(random_state=42))  # Random Forest Classifier
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'rf__n_estimators': [100, 200, 300],  # Number of trees
    'rf__max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'rf__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'rf__min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train the Random Forest model with the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Predict the labels on the test set
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
