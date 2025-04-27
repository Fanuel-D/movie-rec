import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter



csv_path = 'movie.csv'

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return " ".join(tokens)


def load_and_preprocess_data(csv_path):
    """Load movie data and preprocess it"""

    df = pd.read_csv(csv_path)
    title_col = "original_title"
    df['title'] = df[title_col]
    title_to_id = dict(zip(df['title'], df['id']))
    id_to_title = dict(zip(df['id'], df['title']))

    df['cleaned_overview'] = df['overview'].apply(clean_text)
    

    df['original_overview'] = df['overview']
    
    df = df[df['cleaned_overview'].str.len() > 0]

    #save to pickle files 

    f = open("df.pkl", 'wb')
    pickle.dump(df, f )
    f.close()

    # f1 = open("title_to_id.pkl", 'wb')
    # pickle.dump(title_to_id, f1 )
    # f1.close()

    # f1 = open("title_to_id.pkl", 'wb')
    # pickle.dump(title_to_id, f1 )
    # f1.close()


    # f2 = open("id_to_title.pkl", "wb")
    # pickle.dump(id_to_title, f2)
    # f2.close()




def main():
    load_and_preprocess_data(csv_path)
main()