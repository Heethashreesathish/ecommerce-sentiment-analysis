import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Sentiment analysis function using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return {"label": "POSITIVE", "score": min(polarity, 1.0)}
    elif polarity < -0.1:
        return {"label": "NEGATIVE", "score": abs(polarity)}
    else:
        return {"label": "NEUTRAL", "score": 0.0}

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/ecommerce_reviews.csv')
    return df

df = load_data()

# Title
st.title("Ecommerce Review Sentiment Analysis")

# Sidebar
st.sidebar.header("Navigation")
option = st.sidebar.selectbox("Choose an option", ["Dataset Preview", "Single Review Analysis", "Batch Analysis", "Sentiment Distribution"])

if option == "Dataset Preview":
    st.header("Dataset Preview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head(10))

elif option == "Single Review Analysis":
    st.header("Analyze a Single Review")
    user_review = st.text_area("Enter your review:")
    if st.button("Analyze Sentiment"):
        if user_review:
            result = analyze_sentiment(user_review)
            label = result['label']
            score = result['score']
            st.write(f"Sentiment: {label}")
            st.write(f"Confidence: {score:.2f}")
        else:
            st.warning("Please enter a review.")

elif option == "Batch Analysis":
    st.header("Batch Sentiment Analysis")
    if st.button("Analyze All Reviews"):
        with st.spinner("Analyzing..."):
            sentiments = []
            for review in df['review_text']:
                result = analyze_sentiment(review)
                sentiments.append(result['label'])
            df['predicted_sentiment'] = sentiments
            st.success("Analysis Complete!")
            st.dataframe(df[['review_text', 'predicted_sentiment']].head(10))
            # Save to CSV
            df.to_csv('data/processed_reviews_with_predictions.csv', index=False)
            st.info("Processed data saved to data/processed_reviews_with_predictions.csv")

elif option == "Sentiment Distribution":
    st.header("Sentiment Distribution")
    # Assuming we have predicted sentiments
    if 'predicted_sentiment' in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x='predicted_sentiment', data=df, ax=ax)
        ax.set_title('Sentiment Distribution')
        st.pyplot(fig)
    else:
        st.warning("Please run Batch Analysis first to generate sentiment predictions.")