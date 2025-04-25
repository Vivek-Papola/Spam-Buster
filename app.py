import streamlit as st
import pickle
import numpy as np

# Load the vectorizer and model
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Custom CSS for a professional looking UI with color gradients and custom text styles
st.markdown(
    """
    <style>
    /* Gradient background for the entire app */
    .stApp {
        background: linear-gradient(135deg, #ffffff, #e6f2ff);
    }
    /* Styling for the title */
    .title {
        color: #003389;
        font-size: 2.5em;
        font-weight: bold;
    }
    /* Styling for the result text */
    .result {
        font-size: 1.5em;
        font-weight: bold;
        color: #003389;
    }
    /* Styling for the instruction text */
    .instruction {
        color: #003389;
        font-size: 1.2em;
    }
    /* Custom warning style */
    .custom-warning {
        color: #ffffff;
        background-color: #ff6666;
        padding: 10px;
        border-radius: 5px;
        font-size: 1.1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown("<div class='title'>Spam Buster</div>", unsafe_allow_html=True)
st.markdown("<div class='instruction'>Enter a message below and click <b>Predict</b> to check if it's spam or not.</div>", unsafe_allow_html=True)

# User input text area
user_input = st.text_area("Enter Message", height=150)

# Predict button
if st.button("Predict"):
    if user_input.strip():
        # Transform the input text using the loaded vectorizer
        transformed_text = vectorizer.transform([user_input]).toarray()
        # Predict using the loaded model
        prediction = model.predict(transformed_text)
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        st.markdown(f"<div class='result'>Prediction: {result}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='custom-warning'>Please enter a message for prediction!</div>", unsafe_allow_html=True)
