import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import re
import joblib

nltk.download('stopwords')

def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['text'] = df['name'] + ' ' + df['description']
    df['text'] = df['text'].apply(clean_text)
    return df

def train_model(file_path):
    df = load_data(file_path)

    vectorizer = TfidfVectorizer(stop_words=stopwords.words('portuguese'))
    X = vectorizer.fit_transform(df['text'])

    y = df['categories']

    for index, data in enumerate(y):
        if isinstance(data, float):
            y[index] = ''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Acurácia:', accuracy_score(y_test, y_pred))
    print('Relatório de Classificação:\n', classification_report(y_test, y_pred))

    joblib.dump(model, 'modelo_classificador.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

file_path = 'data.csv'
train_model(file_path)
