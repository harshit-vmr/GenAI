import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import streamlit as st
import os

word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

# Load the pre-trained model
model = load_model(os.path.join(os.path.dirname(__file__), 'simple_rnn_imdb.keras'))

def decode_review(review):
    """Decode a review from integers to words."""
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in review])
    return decoded_review

def preprocess_review(review, maxlen=500):
    """Preprocess a review by padding it to the maximum length."""
    words = review.lower().split()
    review = [word_index.get(word, 2) for word in words]
    return sequence.pad_sequences([review], maxlen=maxlen)

### Prediction Function
def predict_review(review):
    """Predict the sentiment of a review."""
    processed_review = preprocess_review(review)
    prediction = model.predict(processed_review)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]


#streamlit app

st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative):")

#user input
user_review = st.text_area("Movie Review")

if st.button("Predict Sentiment"):
    if user_review:
        sentiment, proba = predict_review(user_review)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Prediction proba: {proba}")
    else:
        st.write("Please enter a review to analyze.")
