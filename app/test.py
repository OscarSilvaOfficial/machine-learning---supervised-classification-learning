import joblib

clf = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

test_description = [
    "Curso de marketing: aprenda estratégias de branding, SEO, mídias sociais e análise de dados. Aumente suas habilidades e impulsione seu negócio!"
]

test_tfidf = vectorizer.transform(test_description)

predicted_category = clf.predict(test_tfidf)

print(f'Predicted Category: {predicted_category}')
