import streamlit as st
import pickle
import re

# Load saved model and vectorizer
with open("sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Prediction function
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

# Streamlit UI
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="💬", layout="centered")

st.title("Twitter Sentiment Analysis")
st.write("This app predicts whether a given text expresses Positive, Negative, or Neutral sentiment.")

user_input = st.text_area("Enter a tweet or text:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        sentiment = predict_sentiment(user_input)

        if sentiment == "Positive":
            st.success(f"Predicted Sentiment: {sentiment}")
        elif sentiment == "Negative":
            st.error(f"Predicted Sentiment: {sentiment}")
        else:
            st.info(f"Predicted Sentiment: {sentiment}")

st.markdown("---")
st.write("Model Used: Logistic Regression with TF-IDF Vectorization")
st.write("Accuracy: 79.48%")