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


csv_path = 'movie.csv'
pickle_dir = './pickles'
glove_path = '/data/cs91r-s25/glove/glove.6B.50d.pkl'
EMB_DIM = 50                        # dimensionality of the GloVe vectors
alpha = 0.1                         # TF‑IDF propagation weight
threshold = 0.8                     # cosine‑sim threshold for synonym links


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

    num_movies = max(movie_dict.values()) + 1
    doc_lengths = np.zeros(num_movies)
    word_doc_counts = np.zeros(len(words_dict))

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


def build_and_pickle_similarity(tfidf, words_dict):
    """Create a sparse word-similarity matrix based on GloVe and save it."""
    print('building word-embedding similarity graph ...')

    # load GloVe matrix
    with open(glove_path, 'rb') as f:
        df_embed: pd.DataFrame = pickle.load(
            f)   # index = word, columns = dims

    # create lookup: dense embedding matrix
    V = len(words_dict)
    W_embed = np.zeros((V, EMB_DIM), dtype='float32')

    for word, idx in words_dict.items():
        if word in df_embed.index:
            W_embed[idx] = df_embed.loc[word].values.astype('float32')

    # cosine similarity & thresholding
    sim_mat = cosine_similarity(W_embed, W_embed)
    sim_mat[sim_mat < threshold] = 0.0
    sim_csr = csr_matrix(sim_mat)

    out_path = os.path.join(pickle_dir, 'sim_csr.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(sim_csr, f)
    print(f'similarity graph pickled : {out_path}')

    print('propagating TF-IDF via similarity graph ...')
    propagated = tfidf.dot(sim_csr)
    enhanced = tfidf.multiply(1-alpha) + propagated.multiply(alpha)

    out_path = os.path.join(pickle_dir, 'enhanced_tfidf.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(enhanced, f)
    print(f'enhanced TF-IDF saved: {out_path}')


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
    id_to_title_path = os.path.join(pickle_dir, 'id_to_title_dict.pkl')

    df = load_and_preprocess_data(csv_path)
    movie_dict, words_dict = create_dictionaries(df)
    count_matrix, doc_lengths, word_doc_counts = build_sparse_matrix(
        df, movie_dict, words_dict)
    tfidf_matrix = compute_tfidf_from_counts(
        count_matrix, doc_lengths, word_doc_counts, len(movie_dict))

    movie_dict, words_dict = create_dictionaries(df)
    id_to_title_dict = dict(zip(df['id'], df['title']))

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

    with open(id_to_title_path, 'wb') as f:
        pickle.dump(id_to_title_dict, f)

    words_dict_path = os.path.join(pickle_dir, 'words_dict.pkl')
    with open(words_dict_path, 'rb') as f:
        words_dict = pickle.load(f)
    build_and_pickle_similarity(tfidf_matrix, words_dict)
    print("Data saved to pickles successfully")


if __name__ == '__main__':
    main()
