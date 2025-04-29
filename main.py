import argparse
import sys
import pandas as pd
import pickle
import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import colorama
from colorama import Fore, Back, Style
import pickle
from collections import Counter
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import os


pickle_dir = './pickles'

csv_path = 'movie.csv'

def find_important_words(movie_idx, tfidf_matrix, words_dict, top_n=15):
    """Find important words for a movie based on TF-IDF scores"""
    movie_vector = tfidf_matrix[movie_idx].toarray().flatten()

    word_scores = []
    
    for word, idx in words_dict.items():
        if idx < len(movie_vector) and movie_vector[idx] > 0:
            word_scores.append((word, movie_vector[idx]))
    
    word_scores.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _ in word_scores[:top_n]]

def find_common_important_words(movie1_idx, movie2_idx, tfidf_matrix, words_dict, top_n=10):
    """Find common important words between two movies"""
    movie1_words = set(find_important_words(movie1_idx, tfidf_matrix, words_dict))
    movie2_words = set(find_important_words(movie2_idx, tfidf_matrix, words_dict))
    
    common_words = movie1_words.intersection(movie2_words)
    return list(common_words)[:top_n]


def get_recommendations(movie_id, df, cosine_sim, movie_dict, words_dict, tfidf_matrix, top_n=5):
    """Get top N movie recommendations based on cosine similarity"""

    if movie_id not in movie_dict:
        return {"error": f"Movie ID {movie_id} not found in dataset"}
    
    movie_idx = movie_dict[movie_id]
    movie_title = get_title_from_id(movie_id, df)
    

    sim_scores = list(enumerate(cosine_sim[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    print(sim_scores)
    
    movie_indices = [i[0] for i in sim_scores]
    
    recommendations = []
    for i, movie_idx in enumerate(movie_indices):
        rec_movie_id = None
        for id, idx in movie_dict.items():
            if idx == movie_idx:
                rec_movie_id = id
                break
        
        if rec_movie_id is None:
            continue
            
        movie_row = df[df['id'] == rec_movie_id]
        if len(movie_row) == 0:
            continue
            
        rec_movie_title = movie_row['title'].iloc[0]
        similarity = sim_scores[i][1]
        overview = movie_row['overview'].iloc[0]
        
        #common_words = find_common_important_words(movie_dict[movie_id], movie_idx, tfidf_matrix, words_dict)
        common_words = []
        
        recommendations.append({
            "id": rec_movie_id,
            "title": rec_movie_title,
            "similarity": similarity,
            "overview": overview,
            "common_words": common_words
        })
    
    return {
        "input_movie": movie_title,
        "input_overview": df[df['id'] == movie_id]['overview'].iloc[0],
        "recommendations": recommendations
    }


def search_movie(title_query, df):
    """Search for a movie by title (partial match)"""




    title_query = title_query.lower()
    #print(title_query)
    matches = df[df['title'].str.lower().str.contains(title_query, na=False)]
    print("in here")
    return matches[['id', 'title']]


def inverse_dict(d):
    """Create inverse mapping of a dictionary"""


    inverse = Counter()
    for k, v in d.items():
        inverse[v] = k
    return inverse

def main():
    parser = argparse.ArgumentParser(description="Example using choices in argparse")

    parser.add_argument(
    "mode", 
    choices=["1", "2", "3"],  
    help="Operating mode: train, test, or evaluate"
)
    args = parser.parse_args()
    movie_dict_path = os.path.join(pickle_dir, 'movie_dict.pkl')
    words_dict_path = os.path.join(pickle_dir, 'words_dict.pkl')
    matrix_path = os.path.join(pickle_dir, 'tfidf_matrix.pkl')


    with open("df.pkl", 'rb') as f:
        df = pickle.load(f) 

    # dictionary that maps movie IDs to numerical indices:
    with open(movie_dict_path, 'rb') as f:
            movie_dict = pickle.load(f)
        

    # dictionary that maps vocabulary words to column indices:
    with open(words_dict_path, 'rb') as f:
        words_dict = pickle.load(f)
            
    with open(matrix_path, 'rb') as f:
        tfidf_matrix = pickle.load(f)

    print("done loading all pickles")
    id_to_movie = inverse_dict(movie_dict)

    if args.mode == "1":
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print("\n" + "="*50)
        print("Welcome to Fanuel and Will's Movie Recommendation System!")
        print("="*50)
        print("This system finds similar movies based on overview content.")
        print("Important words that are common between movies are highlighted with colors. Thexe are the words that allow us to find a similarity between movies.")

        while True:
            print("\nEnter a movie title (or 'q' to quit):")
            user_input = input("> ")
            
            if user_input.lower() == 'q':
                break
                
            if not user_input.strip():
                print("Please enter a movie title.")
                continue

            print(f"\nSearching for movies containing '{user_input}'...")
            matches = search_movie(user_input, df)


            if len(matches) == 0:
                print(f"No movies found with title containing '{user_input}'")
                continue
            
            elif len(matches) == 1:
                movie_id = matches['id'].iloc[0]
                movie_title = matches['title'].iloc[0]
                print(f"Found movie: {movie_title}")
            
                # Get recommendations
                results = get_recommendations(movie_id, df, cosine_sim, movie_dict, words_dict, tfidf_matrix)

            
            


    elif args.mode == "2":
        pass
        
        
        

main()