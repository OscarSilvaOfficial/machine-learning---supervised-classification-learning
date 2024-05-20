import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

cred = credentials.Certificate('./serviceAccountKey.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

def get_firestore_data(collection_name):
    collection_ref = db.collection(collection_name)
    docs = collection_ref.stream()
    data = []
    for doc in docs:
        data.append(doc.to_dict())
    print('Data retrieved')
    return data

def extract_data_for_csv(data):
    df = pd.DataFrame(data)
    df.to_csv('./data.csv', index=False)
    print('Data saved on CSV')

extract_data_for_csv(
  data=get_firestore_data('courses_platform')
)