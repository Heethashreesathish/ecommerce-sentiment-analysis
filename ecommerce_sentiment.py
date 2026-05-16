import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sqlalchemy import create_engine
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

# Load data
df = pd.read_csv('data/ecommerce_reviews.csv')

# EDA
print("Data Head:")
print(df.head())
print("\nData Description:")
print(df.describe())

# Visualize rating distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='rating', data=df)
plt.title('Rating Distribution')
plt.savefig('data/rating_distribution.png')
# plt.show()

# Sentiment Analysis using TextBlob
def get_sentiment_textblob(text):
    return TextBlob(text).sentiment.polarity

df['sentiment_polarity'] = df['review_text'].apply(get_sentiment_textblob)
df['sentiment_label'] = df['sentiment_polarity'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

# Visualize sentiment distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='sentiment_label', data=df)
plt.title('Sentiment Distribution')
plt.savefig('data/sentiment_distribution.png')
# plt.show()

# Sentiment Analysis using NLTK VADER
sia = SentimentIntensityAnalyzer()
def get_sentiment_vader(text):
    scores = sia.polarity_scores(text)
    return scores['compound']

df['vader_sentiment'] = df['review_text'].apply(get_sentiment_vader)
df['vader_label'] = df['vader_sentiment'].apply(lambda x: 'positive' if x > 0.05 else 'negative' if x < -0.05 else 'neutral')

# Visualize VADER sentiment
plt.figure(figsize=(8, 5))
sns.countplot(x='vader_label', data=df)
plt.title('VADER Sentiment Distribution')
plt.savefig('data/vader_sentiment_distribution.png')
# plt.show()

# SQL Integration
engine = create_engine('sqlite:///data/reviews.db')
df.to_sql('reviews', engine, if_exists='replace', index=False)

# Query positive reviews
query = "SELECT * FROM reviews WHERE rating >= 4"
positive_reviews = pd.read_sql(query, engine)
print("\nPositive Reviews (Rating >= 4):")
print(positive_reviews)

# Save processed data
df.to_csv('data/processed_reviews.csv', index=False)