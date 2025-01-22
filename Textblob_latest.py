import pandas as pd
from textblob import TextBlob

notebook_path = "/Users/umuteyidogan/Desktop/IGP_Project/crypto_headline_data.csv"

df = pd.read_csv(notebook_path)

crypto_lexicon = {
    'bullish': 2.0,
    'bearish': -2.0,
    'hodl': 1.5,
    'fomo': -1.5,
    'pump': 1.5,
    'dump': -1.5,
    'moon': 2.0,
    'whale': 0.5,
    'altcoin': 0.5,
    'scam': -2.5,
    'rugpull': -2.0,
    'pump and dump': -2.0,
    'moonshot': 2.0,
    'to the moon': 2.0,
    'bear market': -2.0,
    'bull market': 2.0,
    'crypto': 0.5,
    'blockchain': 0.5,
    'yield farming': 1.0,
    'staking': 1.0,
    'token': 0.5,
    'gas fee': -0.5,
    'mining': 0.5,
    'hashrate': 0.5,
    'volatile': -1.5,
    'regulation': -0.5,
    'adoption': 1.5,
    'innovation': 2.0,
    'security': 1.0,
    'fraud': -2.5,
    'hack': -2.0,
    'partnership': 1.5,
    'investment': 1.5,
    'exchange': 0.5,
    'wallet': 0.5,
    'halving': 1.5,
    'funding': 1.0,
    'launch': 1.5,
    'collapse': -2.5,
    'lawsuit': -2.0,
    'profit': 2.0,
    'loss': -2.0,
    'growth': 1.5,
    'decline': -1.5,
    'risk': -1.0,
    'opportunity': 2.0,
    'recovery': 1.5,
    'crash': -2.5,
    'surge': 2.0,
    'plummet': -2.0,
    'rebound': 1.5,
    'stable': 1.0,
    'plunge': -2.0,
    'airdrop': 1.0,
    'bull': 1.5,
    'bear': -1.5,
    'fud': -1.5,
    'rekt': -2.0,
    'satoshi': 0.5,
    'burn': 1.0,
    'mint': 1.0,
    'whitelist': 1.0,
    'blacklist': -1.0,
    'whale': 0.5,
    'staking reward': 1.0,
    'buy the dip': 2.0,
    'sell the news': -1.0,
    'short squeeze': 1.5,
    'margin call': -1.5,
    'paper hands': -1.5,
    'diamond hands': 1.5,
    'moonboy': 1.5,
    'bagholder': -1.5,
    'bear trap': -1.5,
    'bull trap': -1.5,
    'dead cat bounce': -1.5,
    'double top': -1.5,
    'double bottom': 1.5,
    'cup and handle': 1.5,
    'head and shoulders': -1.5,
    'golden cross': 1.5,
    'death cross': -1.5,
    'consolidation': 0.5,
    'take profit': 1.5,
    'breakout': 1.5,
    'breakdown': -1.5,
    'bull flag': 1.5,
    'bear flag': -1.5,
    'buy wall': 1.5,
    'sell wall': -1.5,
    'stop-loss': 0.5,
    'take profit': 1.5,
    'recession': -1.0,
    'economic downturn': -2.0,
    'market correction': -1.5,
    'bull run': 2.0,
    'bear run': -2.0,
    'alt season': 1.5,
    'defi': 0.5,
    'dao': 1.0,
    'dapp': 1.0,
    'nft': 0.5,
    'smart contract': 0.5,
    'ico': 1.5,
    'fud': -2.0,
    'ath': 2.0,
    'atl': -2.0,
    'btc': 0.5,
    'xbt': 0.5,
    'confirmation': 0.5,
    'cold storage': 1.0,
    'consensus': 0.5,
    'cross-chain': 0.5,
    'cryptography': 0.5,
    'decryption': 0.5,
    'dominance': 0.5,
    'double spending': -2.0,
    'dusting attack': -2.0,
    'emission': 0.5,
    'encryption': 0.5,
    'impermanent loss': -1.5,
    'memecoin': -1.0,
    'node': 0.5,
    'order book': 0.5,
    'parachains': 0.5,
    'peer-to-peer (P2P)': 0.5,
    'phishing attack': -2.0,
    'smart contract': 0.5


}

def get_custom_sentiment(text, lexicon):
    if isinstance(text, str):
        words = text.split()
        sentiment_score = 0.0
        for word in words:
            if word in lexicon:
                sentiment_score += lexicon[word]
            else:
                analysis = TextBlob(word).sentiment
                sentiment_score += analysis.polarity
        return sentiment_score / len(words) if words else 0
    return 0

df['Sentiment'] = df['Headline'].apply(lambda x: get_custom_sentiment(x, crypto_lexicon))

# Assign sentiment labels
def assign_label(score):
    if score > 0:
        return 1
    elif score < 0:
        return -1
    else:
        return 0

df['Sentiment_Label'] = df['Sentiment'].apply(assign_label)

print(df)

 #Aggregating daily sentiment
df['Published date'] = pd.to_datetime(df['Published date']).dt.date  # Ensure the date column is in datetime format
daily_sentiment = df.groupby('Published date').agg(
    Positive_Count=('Sentiment_Label', lambda x: (x == 1).sum()),
    Negative_Count=('Sentiment_Label', lambda x: (x == -1).sum()),
    Neutral_Count=('Sentiment_Label', lambda x: (x == 0).sum())
).reset_index()

# Calculate percentages
daily_sentiment['Total'] = daily_sentiment['Positive_Count'] + daily_sentiment['Negative_Count'] + daily_sentiment['Neutral_Count']
daily_sentiment['Positive_Percentage'] = (daily_sentiment['Positive_Count'] / daily_sentiment['Total']) * 100
daily_sentiment['Negative_Percentage'] = (daily_sentiment['Negative_Count'] / daily_sentiment['Total']) * 100
daily_sentiment['Neutral_Percentage'] = (daily_sentiment['Neutral_Count'] / daily_sentiment['Total']) * 100

print(daily_sentiment)

# Save the results

output_path = "/Users/umuteyidogan/Desktop/IGP_Project/Crypto_Sentiment_Analysis.xlsx"
df.to_excel(output_path, index=False)

print(f"Analysis results have been saved to {output_path}")

# Save daily sentiment to a new file
daily_sentiment_output_path = "/Users/umuteyidogan/Desktop/IGP_Project/Daily_Sentiment_Analysis.xlsx"
daily_sentiment.to_excel(daily_sentiment_output_path, index=False)
print(f"Daily sentiment resultsHere's the updated code that includes the columns for daily sentiment aggregation and calculates the percentages of positive, negative, and neutral sentiments:")

'''
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(df['Sentiment'], bins=20, edgecolor='k', alpha=0.7)
plt.title('Sentiment Analysis Distribution')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()  

'''