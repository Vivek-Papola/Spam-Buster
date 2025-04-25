# Spam-Buster : SMS Spam Detection App

This repository implements an end-to-end SMS spam detection pipeline, from data cleaning and exploratory analysis, through NLP-based preprocessing and model training, to deployment as a Streamlit web app.

## Steps for Spam Detection

1. **Data Cleaning**  
2. **Performing EDA**  
3. **Text Preprocessing**  
4. **Model Building**  
5. **Evaluation**  
6. **Improvement & Tuning**  
7. **Web App Development**  
8. **Deployment**

## Features

- **Data Cleaning & Deduplication**: Load raw CSV, drop unnecessary columns, handle nulls & duplicates  
- **Exploratory Data Analysis**: Visualize class imbalance, message lengths, word & sentence distributions  
- **Text Preprocessing**: Tokenization, stop-word removal, stemming & alphanumeric filtering with NLTK  
- **Feature Engineering**: Vectorize text via TF–IDF (max_features=3000) and optional message-length features  
- **Multiple Classifiers**: Train & compare GaussianNB, MultinomialNB, BernoulliNB, SVM, Decision Tree, Random Forest, AdaBoost, Bagging, ExtraTrees, Gradient Boosting & XGBoost  
- **Model Evaluation**: Compute accuracy & precision, display performance comparison charts  
- **Model Persistence**: Serialize TF–IDF vectorizer & best classifier with `pickle` for real-time inference  
- **Interactive Streamlit UI**: Input custom SMS messages and get “Spam” or “Ham” predictions instantly  

## Tech Stack

- **Python 3.8+**  
- **Pandas** for data manipulation  
- **NLTK** for NLP preprocessing  
- **scikit-learn** for vectorization & modeling  
- **Matplotlib & Seaborn** for visualization  
- **Streamlit** for web app  
- **pickle** for model serialization  

## Installation

Clone the repo  
```bash
git clone https://github.com/your-username/sms-spam-detector.git
cd sms-spam-detector
