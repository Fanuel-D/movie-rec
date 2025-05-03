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
import editdistance


HIGHLIGHT_COLORS = [
    Fore.RED,
    Fore.GREEN,
    Fore.BLUE,
    Fore.MAGENTA,
    Fore.CYAN
]


pickle_dir = './pickles'

csv_path = 'movie.csv'


# def edits1(word):
#     "All edits that are one edit away from `word`."
#     letters = 'abcdefghijklmnopqrstuvwxyz'
#     splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
#     deletes = [L + R[1:] for L, R in splits if R]
#     transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
#     replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
#     inserts = [L + c + R for L, R in splits for c in letters]
#     return set(deletes + transposes + replaces + inserts)


# def edits2(word):
#     "All edits that are two edits away from `word`."
#     # return (e2 for e1 in edits1(word) for e2 in edits1(e1))
#     lst = []
#     for e1 in edits1(word):
#         for e2 in edits1(e1):
#             lst.append(e2)

#     return lst


def search_movie(title_query, df):
    """Search for a movie by title (partial match)"""

    title_query = title_query.lower()
    # print(title_query)
    matches = df[df['title'].str.lower().str.contains(title_query, na=False)]
    print("in here")
    return matches[['id', 'title']]


def search_movie2(title_query, df):
    """Search for a movie by title (partial match)"""

    title_query = title_query.lower()

    df_list = []
    matches1 = df[df['title'].str.lower().str.contains(title_query, na=False)]
    for word in edits:
        df_list.append(
            df[df['title'].str.lower().str.contains(word, na=False)])

    df_list.append(matches1)

    matches = pd.concat(df_list)
    # print(matches, type(matches))
    print("in here")
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


def find_common_important_words(movie1_idx, movie2_idx, tfidf_matrix, words_dict, top_n=10):
    """Find common important words between two movies"""

    movie1_words = set(find_important_words(
        movie1_idx, tfidf_matrix, words_dict))
    movie2_words = set(find_important_words(
        movie2_idx, tfidf_matrix, words_dict))

    common_words = movie1_words.intersection(movie2_words)

    return list(common_words)[:top_n]


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

        common_words = find_common_important_words(
            movie_dict[movie_id], movie_idx, tfidf_matrix, words_dict)

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


def inverse_dict(d):
    """Create inverse mapping of a dictionary"""

    inverse = Counter()
    for k, v in d.items():
        inverse[v] = k
    return inverse


def main():
    parser = argparse.ArgumentParser(
        description="Example using choices in argparse")

    parser.add_argument(
        "mode",
        choices=["1", "2", "3"],
        help="Operating mode: train, test, or evaluate"
    )
    args = parser.parse_args()
    movie_dict_path = os.path.join(pickle_dir, 'movie_dict.pkl')
    words_dict_path = os.path.join(pickle_dir, 'words_dict.pkl')
    matrix_path = os.path.join(pickle_dir, 'tfidf_matrix.pkl')
    id_to_tile_path = os.path.join(pickle_dir, 'id_to_title_dict.pkl')

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

    with open(id_to_tile_path, 'rb') as f:
        id_to_title = pickle.load(f)

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
            matches = search_movie(user_input, df, id_to_title)

            if len(matches) == 0:
                print(f"No movies found with title containing '{user_input}'")
                continue

            elif len(matches) == 1:
                movie_id = matches['id'].iloc[0]
                movie_title = matches['title'].iloc[0]
                print(f"Found movie: {movie_title}")

                results = get_recommendations(
                    movie_id, df, cosine_sim, movie_dict, words_dict, tfidf_matrix)
                print_recommendations(results)

            else:
                print(f"Found {len(matches)} movies:")
                for i, (_, row) in enumerate(matches.iterrows(), 1):
                    print(f"{i}. {row['title']}")

                print("\nEnter the number of your selection:")

                selection = int(input("> ")) - 1
                if 0 <= selection < len(matches):
                    movie_id = matches['id'].iloc[selection]
                    movie_title = matches['title'].iloc[selection]
                    print(f"\nGetting recommendations for '{movie_title}'...")

                    # Get recommendations
                    results = get_recommendations(
                        movie_id, df, cosine_sim, movie_dict, words_dict, tfidf_matrix)

                    if "error" in results:
                        print(f"Error: {results['error']}")
                    else:
                        print_recommendations(results)
                else:
                    print("Invalid selection.")

    elif args.mode == "2":
        pass


if __name__ == '__main__':
    main()
