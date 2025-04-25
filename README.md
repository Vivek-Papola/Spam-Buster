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

## Dataset

The project uses the SMS Spam Collection dataset, comprising 5,574 English SMS messages labeled as “ham” or “spam” extracted from kaggle.

## Installation

Clone the repo

```bash
  git clone https://github.com/<your-username>/sms-spam-detector.git  
  cd sms-spam-detector  
```
Create and activate a virtual environment
```bash
  python3 -m venv venv  
  source venv/bin/activate   # Mac/Linux  
  venv\Scripts\activate      # Windows  
```
Install dependencies
```bash
  pip install -r requirements.txt  
```
The app will open in your default browser at localhost:8501
## Usage

   1. Run the data pipeline & model training.

      ```bash
      python train.py   # cleans data, trains models, saves best model & vectorizer  
      ```

   2. Launch the streamlit app
      ```bash
      streamlit run app.py  
      ```
   3. Interact with the UI

      - Paste or type an SMS into the text box

      - Click Predict to see “Spam” or “Ham (Not Spam)” and related probability scores
  
## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository

2. Create a feature branch(`git checkout -b feature/MyFeature`)

3. Commit your changes(`git commit -m "Add MyFeature"`)

4. Push to the branch(`git push origin feature/MyFeature`)

5. Open a Pull Request

Please adhere to this project's `code of conduct`.


## License

[MIT](https://choosealicense.com/licenses/mit/) - 
This project is licensed under the MIT License – see the LICENSE file for details.

## Authors

- [@vivek-papola](https://www.github.com/vivek-papola)

Feel free to star ⭐ the repo if you find Spam-Buster useful!
