# Bitcoin Price Prediction Using Sentiment Analysis on News Headlines

## **Overview**

This project leverages sentiment analysis and machine learning to predict Bitcoin price movements based on newspaper headlines. By analyzing the sentiment of over 15,000 Bitcoin-related headlines, the study explores the relationship between public sentiment and Bitcoin's market dynamics. The predictions classify whether to buy, sell, or hold Bitcoin, using advanced data processing and machine learning techniques.

---

## **Features**

- **Sentiment Analysis**: Uses TextBlob and VADER libraries to extract sentiment scores from newspaper headlines.
- **Machine Learning Models**: Implements Support Vector Machine (SVM), Logistic Regression, and Random Forest for classification.
- **Technical Indicators**: Incorporates indicators like Moving Average Convergence Divergence (MACD), Relative Strength Index (RSI), and Bollinger Bands for enhanced predictions.
- **Data Visualization**: Provides insights into sentiment trends and their correlation with Bitcoin price movements using Matplotlib and Seaborn.
- **Trading Simulation**: Tests model efficacy with a simulated trading strategy.

---

## **Dataset**

- **Bitcoin Price Data**: Collected from [CoinCodex](https://coincodex.com).
- **Newspaper Headlines**: Sourced from Nexus and Kaggle datasets (e.g., *Crypto News Headlines & Market Prices by Date*).

---

## **Methodology**

1. **Data Preprocessing**:
   - Removed non-English headlines, stopwords, and duplicates.
   - Applied lemmatization and noise reduction using NLTK and TextBlob.
2. **Sentiment Analysis**:
   - Calculated polarity using TextBlob and compound scores using VADER.
   - Aggregated daily sentiment trends for comparison.
3. **Feature Engineering**:
   - Integrated technical indicators using the TA-Lib library.
   - Standardized features for consistent model performance.
4. **Model Development**:
   - Used scikit-learn to implement and optimize SVM, Logistic Regression, and Random Forest models.
5. **Evaluation**:
   - Measured accuracy, precision, recall, and F1-score.
   - Conducted a trading simulation for practical validation.

---

## **Results**

- **Correlation Insights**: Explored relationships between sentiment trends and Bitcoin price changes.
- **Model Performance**: Achieved significant insights using SVM, Logistic Regression, and Random Forest, with varied accuracy levels.
- **Trading Simulation**: Demonstrated the model's ability to outperform random trading strategies in profitability.

---
## **Future Work**

- Expand data sources to include more diverse news outlets.
- Explore advanced NLP techniques for crypto-specific sentiment analysis.
- Integrate deep learning models like LSTMs and GRUs for improved accuracy.

---

## **Contributors**

- **Nasir Dalal**
- **Idil Ersudas**
- **Umut Eyidogan**
- **Josie Graham**
- **Elizabeth Ofosu**


