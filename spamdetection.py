# Steps for spam detection:-
# 1. Data Cleaning
# 2. Performing EDA
# 3. Text preprocessing
# 4. Model building
# 5. Evaluation
# 6. Improve
# 7. Website
# 8. Deploy

import numpy as np
import pandas as pd

df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df.sample(5)

# 1. Data Cleaning
df.shape
df.info()

# drop last 3 cols
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

df.sample(5)

# renaming the cols
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
df.sample(5)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['target'] = encoder.fit_transform(df['target'])
df.head()

# missing values
df.isnull().sum()

# duplicate values
df.duplicated().sum()
df = df.drop_duplicates(keep = 'first')
df.duplicated().sum()
df.shape

# 2. Performing EDA
df.head()
df['target'].value_counts()

import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
plt.show()

# data is imbalanced
import nltk
nltk.download('punkt')
nltk.download('stopwords')

df['num_characters'] = df['text'].apply(len)
df.head()

# number of words
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df.head()

df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
df.head()

df[['num_characters', 'num_words', 'num_sentences']].describe()
df[df['target']==0][['num_characters', 'num_words', 'num_sentences']].describe()
df[df['target']==1][['num_characters', 'num_words', 'num_sentences']].describe()

import seaborn as sns

plt.figure(figsize=(12, 6))
sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'], color='red')

plt.figure(figsize=(12, 6))
sns.histplot(df[df['target']==0]['num_words'])
sns.histplot(df[df['target']==1]['num_words'], color='red')

sns.pairplot(df, hue='target')

numeric_df = df.select_dtypes(include='number')

# compute the correlation matrix
correlation_matrix = numeric_df.corr()

# create a heatmap with annotations
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()

# 3. Data Preprocessing
from nltk.corpus import stopwords
import string

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('dancing')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

transform_text('I love learning Machine Learning. What about you?')
df['transformed_text'] = df['text'].apply(transform_text)
df.head()

from wordcloud import WordCloud
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

spam_wc = wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))
plt.imshow(spam_wc)

ham_wc = wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))
plt.imshow(ham_wc)

df.head()

spam_corpus = []
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
len(spam_corpus)

from collections import Counter
most_common_words = Counter(spam_corpus).most_common(30)

# create a dataframe
df_most_common = pd.DataFrame(most_common_words, columns=['word', 'count'])

# create the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='word', y='count', data=df_most_common)
plt.xticks(rotation = 'vertical')
plt.show()

ham_corpus = []
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)
len(ham_corpus)

from collections import Counter
most_common_words = Counter(ham_corpus).most_common(30)

# create a dataframe
df_most_common = pd.DataFrame(most_common_words, columns=['word', 'count'])

# create the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='word', y='count', data=df_most_common)
plt.xticks(rotation = 'vertical')
plt.show()

# 4. Model Building
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['transformed_text']).toarray()
# X = np.hstack((X, df['num_characters'].values.reshape(-1, 1)))
X.shape

y = df['target'].values
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))

mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))

bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

svc = SVC(kernel = 'sigmoid', gamma = 1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)

clfs = {
    'SVC' : svc,
    'KN' : knc,
    'NB' : mnb,
    'DT' : dtc,
    'LR' : lrc,
    'RF' : rfc,
    'AdaBoost' : abc,
    'BgC' : bc,
    'ETC' : etc,
    'GBDT' : gbdt,
    'xgb' : xgb
}

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, precision

train_classifier(svc, X_train, y_train, X_test, y_test)

accuracy_scores = []
precision_scores = []

for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(clf, X_train, y_train, X_test, y_test)

    print("For ", name)
    print("Accuracy - ", current_accuracy)
    print("Precision - ", current_precision)

    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

performance_df = pd.DataFrame({
    'Algorithm':clfs.keys(),
    'Accuracy':accuracy_scores,
    'Precision':precision_scores
}).sort_values('Precision',ascending=False)
print(performance_df)

performance_df1 = pd.melt(performance_df, id_vars="Algorithm")
performance_df1

sns.catplot(x = 'Algorithm', y = 'value',
            hue = 'variable', data = performance_df1,
            kind = 'bar', height = 5)
plt.ylim(0.5, 1.0)
plt.xticks(rotation = 'vertical')
plt.show()

# Model Improvisation
# 1. Change the max features parameter of TfIdf

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores})
performance_df.merge(temp_df, on='Algorithm')

import pickle
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))

import streamlit as st
import pickle
import numpy as np

# Load the vectorizer and model
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Custom CSS for a professional looking UI with color gradients
st.markdown(
    """
    <style>
    /* Gradient background for the entire app */
    .stApp {
        background: linear-gradient(135deg, #ffffff, #e6f2ff);
    }
    /* Styling for the title */
    .title {
        color: #003366;
        font-size: 2.5em;
        font-weight: bold;
    }
    /* Styling for the result text */
    .result {
        font-size: 1.5em;
        font-weight: bold;
        color: #003366;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown("<div class='title'>Spam Detection App</div>", unsafe_allow_html=True)
st.write("Enter a message below and click **Predict** to check if it's spam or not.")

# User input text area
user_input = st.text_area("Enter Message", height=150)

# Predict button
if st.button("Predict"):
    if user_input.strip():
        # Transform the input text using the loaded vectorizer
        transformed_text = vectorizer.transform([user_input]).toarray()
        # Predict using the loaded model
        prediction = model.predict(transformed_text)
        result = "Spam" if prediction[0] == 1 else "Ham (Not Spam)"
        st.markdown(f"<div class='result'>Prediction: {result}</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a message for prediction!")
