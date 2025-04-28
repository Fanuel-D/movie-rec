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
        
        

main()