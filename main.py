import argparse
import sys
import os
import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd
import nltk
import editdistance
from nltk.corpus import stopwords
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import colorama
from colorama import Fore, Back, Style


HIGHLIGHT_COLORS = [Fore.RED, Fore.GREEN, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]
pickle_dir = './pickles'


def inverse_dict(d):
    """Create inverse mapping of a dictionary"""
    inverse = Counter()
    for k, v in d.items():
        inverse[v] = k
    return inverse


def search_movie(title_query, df, id_to_tile):
    """Search for a movie by title (partial match)"""

    title_query = title_query.lower()

    titles = id_to_tile.values()
    possible_matches = []
    for title in titles:
        edit_dist = editdistance.eval(str(title), title_query)
        if edit_dist < 6:
            possible_matches.append((title, edit_dist))

    sorted_possible_matches_list = list(
        sorted(possible_matches, key=lambda x: x[1]))[:20]

    possible_titles = [title for title, dist in sorted_possible_matches_list]

    df_list = [
        df[df['title'].str.lower().str.contains(poss_title.lower(), na=False)]
        for poss_title in possible_titles
    ]
    matches = pd.concat(df_list, ignore_index=True).drop_duplicates()

    return matches[['id', 'title']]


def get_title_from_id(movie_id, df):

    movie_row = df[df['id'] == movie_id]
    if len(movie_row) > 0:
        return movie_row['title'].iloc[0]
    return None


def highlight_words_in_text(text, words_to_highlight):

    text = str(text)
    word_pattern = r'\b(?:' + '|'.join(re.escape(word)
                                       for word in words_to_highlight) + r')\b'

    def colorize(match):
        word = match.group(0)
        word_index = next((i for i, w in enumerate(
            words_to_highlight) if w.lower() == word.lower()), 0)
        color_index = word_index % len(HIGHLIGHT_COLORS)
        return f"{HIGHLIGHT_COLORS[color_index]}{word}{Style.RESET_ALL}"
    highlighted_text = re.sub(word_pattern, colorize,
                              text, flags=re.IGNORECASE)

    return highlighted_text


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

    movie1_words = set(find_important_words(
        movie1_idx, tfidf_matrix, words_dict))
    movie2_words = set(find_important_words(
        movie2_idx, tfidf_matrix, words_dict))

    common_words = movie1_words.intersection(movie2_words)
    return list(common_words)[:top_n]


def print_recommendations(results):
    """Print recommendations with highlighted text"""
    input_movie = results['input_movie']
    input_overview = results['input_overview']

    print(f"\n{'='*80}")
    print(f"INPUT MOVIE: {Fore.YELLOW}{input_movie}{Style.RESET_ALL}")
    print(f"Overview: {input_overview}")
    print(f"{'='*80}\n")

    print(f"Top 5 similar movies to '{input_movie}':")

    for i, rec in enumerate(results['recommendations'], 1):
        title = rec['title']
        if len(title) > 50:
            title = title[:47] + "..."

        highlighted_overview = highlight_words_in_text(
            rec['overview'], rec['common_words'])

        print(f"\n{'-'*80}")
        print(
            f"{i}. {Fore.CYAN}{title}{Style.RESET_ALL} (Similarity: {Fore.YELLOW}{rec['similarity']:.4f}{Style.RESET_ALL})")

        if rec['common_words']:
            print("\nKey similar words:")
            for j, word in enumerate(rec['common_words']):
                color_index = j % len(HIGHLIGHT_COLORS)
                print(
                    f"{HIGHLIGHT_COLORS[color_index]}{word}{Style.RESET_ALL}", end=" ")
            print("\n")

        print("Overview:")
        print(highlighted_overview)
        print(f"{'-'*80}")


def cosine_row(idx: int, matrix):
    """
    Cosine similarity of one row (movie idx) against the whole sparse matrix.
    Returns a 1-D NumPy array of length N.
    """
    return cosine_similarity(matrix[idx], matrix).flatten()


def get_recommendations(
        movie_id: int,
        df: pd.DataFrame,
        enhanced_matrix,
        movie_dict: dict,
        words_dict: dict,
        tfidf_matrix,
        top_n: int = 5
):
    """Return top-N movie recommendations for a given `movie_id`."""

    if movie_id not in movie_dict:
        return {"error": f"Movie ID {movie_id} not found in dataset"}

    idx = movie_dict[movie_id]          # row index in the matrices
    input_title = get_title_from_id(movie_id, df)

    # similarity vector
    sims = cosine_row(idx, enhanced_matrix)

    # skip the movie itself (similarity == 1 at idx) and pick top‑N
    best_indices = np.argsort(-sims)[1:top_n+1]

    # build results
    inv_map = {v: k for k, v in movie_dict.items()}   # idx to movie_id
    recommendations = []

    for j in best_indices:
        rec_id = inv_map[j]
        row = df[df['id'] == rec_id].iloc[0]

        recommendations.append({
            "id": rec_id,
            "title": row['title'],
            "similarity": float(sims[j]),
            "overview": row['overview'],
            "common_words": find_common_important_words(idx, j,
                                                        enhanced_matrix,
                                                        words_dict)
        })

    return {
        "input_movie": input_title,
        "input_overview": df[df['id'] == movie_id]['overview'].iloc[0],
        "recommendations": recommendations
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['1', '2', '3'])
    args = parser.parse_args()

    # Load data
    df = pickle.load(open('df.pkl', 'rb'))
    movie_dict = pickle.load(
        open(os.path.join(pickle_dir, 'movie_dict.pkl'), 'rb'))
    words_dict = pickle.load(
        open(os.path.join(pickle_dir, 'words_dict.pkl'), 'rb'))
    tfidf = pickle.load(
        open(os.path.join(pickle_dir, 'tfidf_matrix.pkl'), 'rb'))
    id_to_title = pickle.load(
        open(os.path.join(pickle_dir, 'id_to_title_dict.pkl'), 'rb'))
    enhanced = pickle.load(
        open(os.path.join(pickle_dir, 'enhanced_tfidf.pkl'), 'rb'))
    inv_map = inverse_dict(movie_dict)
    print("Loaded data and TF-IDF")

    if args.mode == '1':
        print("\n" + "="*50)
        print("Welcome to Fanuel and Will's Movie Recommendation System!")
        print("="*50)
        print("This system finds similar movies based on overview content.")
        print("Important words that are common between movies are highlighted with colors. There are the words that allow us to find a similarity between movies.")

        while True:
            inp = input("\nEnter a movie title (or 'q' to quit): ")
            if inp.lower() == 'q':
                break
            if not inp.strip():
                continue
            matches = search_movie(inp, df, id_to_title)
            if matches.empty:
                print(f"No matches for '{inp}'")
                continue
            if len(matches) == 1:
                movie_id = matches['id'].iloc[0]
                movie_title = matches['title'].iloc[0]
                print(f"Found movie: {movie_title}")

                results = get_recommendations(
                    movie_id, df, enhanced, movie_dict, words_dict, tfidf)
                print_recommendations(results)
            else:
                print(f"Found {len(matches)} movies:")
                print(f"Printing at most 10 or less matches ")
                for i, (_, row) in enumerate(matches.iterrows(), 1):
                    if i == 11:
                        break
                    print(f"{i}. {row['title']}")

                print("\nEnter the number of your selection:")

                selection = int(input("> ")) - 1
                if 0 <= selection < len(matches):
                    movie_id = matches['id'].iloc[selection]
                    movie_title = matches['title'].iloc[selection]
                    print(f"\nGetting recommendations for '{movie_title}'...")

                    # Get recommendations
                    results = get_recommendations(
                        movie_id, df, enhanced, movie_dict, words_dict, tfidf)
                    if "error" in results:
                        print(f"Error: {results['error']}")
                    else:
                        print_recommendations(results)
                else:
                    print("Invalid selection.")


if __name__ == '__main__':
    main()
