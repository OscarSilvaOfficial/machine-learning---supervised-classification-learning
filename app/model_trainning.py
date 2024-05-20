import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import ast
import joblib

def drop_unused_columns(df: pd.DataFrame):
    columns = 'courseItemsNumber,id,numberOfArticles,readingTime,updatedAt,numberOfTopics,createdAt,version'
    for column in columns.split(','): 
        del df[column]

def remove_nan_from_array(array: list):
    return [data for data in array if not isinstance(data, float)]

def save_model(clf: RandomForestClassifier):
    joblib.dump(clf, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    print("Model saved")

# ---------------- Execution ------------------- #

df = pd.read_csv('./data.csv')
drop_unused_columns(df)

print("Number of courses: ", len(df))
print("Columns: ", list(df.columns))

all_categories = []
for categories in df['categories'].to_list():
    if not isinstance(categories, float):
        categories_list = ast.literal_eval(categories)
        for category in categories_list:
            all_categories.append(category)

print("Categories: ", all_categories)

df['description'] = df['description'].fillna('')
df['name'] = df['name'].fillna('')

labels = list(df['categories'].values)
features = list(df['description'].values)

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

save_model(clf)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
