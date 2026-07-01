import streamlit as st
import pickle
import re
import pandas as pd

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


def find_text_column(columns):
    preferred_columns = [
        "text",
        "tweet",
        "tweet_text",
        "full_text",
        "content",
        "body",
    ]
    normalized = {column.lower().strip(): column for column in columns}
    for name in preferred_columns:
        if name in normalized:
            return normalized[name]
    return None

# Streamlit UI
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="💬", layout="centered")

st.title("Twitter Sentiment Analysis")
st.write("This app predicts whether a given text expresses Positive, Negative, or Neutral sentiment.")

single_tab, batch_tab = st.tabs(["Single Text", "CSV Batch"])

with single_tab:
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

with batch_tab:
    st.write("Upload a CSV with tweet text. TweetClaw exports work when they include a text-like column.")
    uploaded_file = st.file_uploader("CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        text_column = find_text_column(data.columns)

        if text_column is None:
            st.warning("Add a text, tweet, tweet_text, full_text, content, or body column.")
        else:
            results = data.copy()
            results["predicted_sentiment"] = results[text_column].fillna("").map(predict_sentiment)
            st.dataframe(results[[text_column, "predicted_sentiment"]].head(25), use_container_width=True)
            st.download_button(
                "Download Predictions",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="tweet_sentiment_predictions.csv",
                mime="text/csv",
            )

st.markdown("---")
st.write("Model Used: Logistic Regression with TF-IDF Vectorization")
st.write("Accuracy: 79.48%")
