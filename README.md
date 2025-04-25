# Spam-Buster : SMS Spam Detection App

This repository implements an end-to-end SMS spam detection pipeline, starting from data ingestion and cleaning, through exploratory data analysis (EDA) and NLP-based text preprocessing, to model training, evaluation, and deployment as a Streamlit web app. It uses the UCI SMS Spam Collection dataset, demonstrates multiple classifiers (e.g., Naive Bayes, SVM, Random Forest), and persists the best model with pickle for real-time inference

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

- **Python 3.8+:** core language 
- **Pandas:** for data manipulation  
- **NLTK:** for NLP preprocessing  
- **scikit-learn:** for vectorization & modeling  
- **Matplotlib & Seaborn:** for visualization  
- **Streamlit:** for web app  
- **pickle:** for model serialization  

## Installation

Clone the repo  
```bash
git clone https://github.com/your-username/sms-spam-detector.git
cd sms-spam-detector
```
