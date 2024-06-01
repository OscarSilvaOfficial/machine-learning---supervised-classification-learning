import joblib
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

nltk.download('stopwords')

def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

def load_model_and_vectorizer(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_category(model, vectorizer, nome, descricao):
    text = clean_text(nome + ' ' + descricao)
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]

if __name__ == "__main__":
    nome_curso = ""
    descricao_curso = ""

    model_path = 'modelo_classificador.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'

    # Carregar modelo e vetorizer
    model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)

    # Predizer a categoria do novo curso
    categoria = predict_category(model, vectorizer, nome_curso, descricao_curso)
    print(f'A categoria prevista para o curso "{nome_curso}" é: {categoria}')
