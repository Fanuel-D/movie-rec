from collections import Counter
from nltk.corpus import stopwords
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, coo_matrix
import re
import nltk
import os
# nltk.download('punkt_tab')


csv_path = 'movie.csv'


def compute_tfidf_from_counts(count_matrix, doc_lengths, word_doc_counts, num_docs):
    """Convert count matrix to TF-IDF matrix"""

    tfidf_matrix = count_matrix.copy()

    row_indices, col_indices = tfidf_matrix.nonzero()
    for i, j in zip(row_indices, col_indices):
        if doc_lengths[i] > 0:
            tfidf_matrix[i, j] = tfidf_matrix[i, j] / doc_lengths[i]

    for j in range(tfidf_matrix.shape[1]):
        if word_doc_counts[j] > 0:

            idf = np.log(num_docs / word_doc_counts[j])

            tfidf_matrix[:, j] *= idf

    return tfidf_matrix


def create_dictionaries(df):
    """Create dictionaries mapping movie IDs and words to indices"""

    movie_dict = {}
    for idx, movie_id in enumerate(df['id']):
        movie_dict[movie_id] = idx

    all_words = set()
    for overview in df['overview']:
        if pd.isna(overview):
            continue
        tokens = clean_text(overview)
        all_words.update(tokens)

    words_dict = {}
    for idx, word in enumerate(sorted(all_words)):
        words_dict[word] = idx

    print(f"Found {len(movie_dict)} movies and {len(words_dict)} unique words")
    return movie_dict, words_dict


def build_sparse_matrix(df, movie_dict, words_dict):
    """Build a sparse matrix directly from the movie overviews"""

    data = []
    rows = []
    cols = []

    # token counts for each overview for TF-IDF calculation later
    num_movies = max(movie_dict.values()) + 1
    # num_movies = max(movie_dict.values())
    doc_lengths = np.zeros(num_movies)
    word_doc_counts = np.zeros(len(words_dict))

    # each mvie
    for idx, row in df.iterrows():
        movie_id = row['id']
        overview = row['overview']

        if pd.isna(overview) or movie_id not in movie_dict:
            continue

        movie_idx = movie_dict[movie_id]

        tokens = clean_text(overview)
        word_counts = Counter(tokens)

        for word, count in word_counts.items():
            if word in words_dict:
                word_idx = words_dict[word]
                rows.append(movie_idx)
                cols.append(word_idx)
                data.append(count)
                word_doc_counts[word_idx] += 1

        if movie_idx < len(doc_lengths):
            doc_lengths[movie_idx] = len(tokens)

    matrix = coo_matrix((data, (rows, cols)),
                        shape=(num_movies, len(words_dict)),
                        dtype=np.float32)

    matrix_csr = matrix.tocsr()
    return matrix_csr, doc_lengths, word_doc_counts


def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = str(text).lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [
        word for word in tokens if word not in stop_words and len(word) > 2]

    return tokens


def load_and_preprocess_data(csv_path):
    """Load movie data and preprocess it"""

    df = pd.read_csv(csv_path)
    title_col = 'original_title'

    df['title'] = df[title_col]

    df = df[df['overview'].notna()]

    print(f"Length of df is {len(df)}")

    f = open("df.pkl", 'wb')
    pickle.dump(df, f)
    f.close()

    return df


def main():
    csv_path = 'movie.csv'
    pickle_dir = './pickles'
    movie_dict_path = os.path.join(pickle_dir, 'movie_dict.pkl')
    words_dict_path = os.path.join(pickle_dir, 'words_dict.pkl')
    matrix_path = os.path.join(pickle_dir, 'tfidf_matrix.pkl')

    df = load_and_preprocess_data(csv_path)
    movie_dict, words_dict = create_dictionaries(df)
    count_matrix, doc_lengths, word_doc_counts = build_sparse_matrix(
        df, movie_dict, words_dict)
    tfidf_matrix = compute_tfidf_from_counts(
        count_matrix, doc_lengths, word_doc_counts, len(movie_dict))

    movie_dict, words_dict = create_dictionaries(df)

    # Build sparse count matrix
    count_matrix, doc_lengths, word_doc_counts = build_sparse_matrix(
        df, movie_dict, words_dict)

    # Convert to TF-IDF
    tfidf_matrix = compute_tfidf_from_counts(
        count_matrix, doc_lengths, word_doc_counts, len(movie_dict))

    # Save data to pickles
    print("Saving data to pickles...")
    with open(movie_dict_path, 'wb') as f:
        pickle.dump(movie_dict, f)

    with open(words_dict_path, 'wb') as f:
        pickle.dump(words_dict, f)

    with open(matrix_path, 'wb') as f:
        pickle.dump(tfidf_matrix, f)

    print("Data saved to pickles successfully")


if __name__ == '__main__':
    main()
