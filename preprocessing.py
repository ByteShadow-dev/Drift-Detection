import pandas as pd
import numpy as np
import os
from typing import Any, Tuple, List

def genre_to_binary_vector(genre_string: str, all_genres: list) -> np.ndarray:
    """
    Converts a pipe-separated genre string into a fixed-length binary vector.
    Example: 'Action|Sci-Fi' -> [1, 0, 0, 1, 0...]
    
    No probabilities or smoothing are applied at this stage.
    """
    if pd.isna(genre_string) or genre_string == '(no genres listed)':
        movie_genres = []
    else:
        movie_genres = genre_string.split('|')
    
    count_vector = np.array([1.0 if g in movie_genres else 0.0 for g in all_genres])
    return count_vector

def load_and_prepare(dataset_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads ratings and movies, merges them, and generates a genre vector for every interaction.
    Sorts result chronologically by user.
    """
    # Load MovieLens ratings and genres
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    genres = pd.read_csv(os.path.join(base_dir, dataset_path, 'movies.csv'))
    ratings = pd.read_csv(os.path.join(base_dir, dataset_path, 'ratings.csv'))
    
    merged_data = ratings.merge(genres[['movieId', 'genres']], on='movieId', how='left')
    merged_data['date'] = pd.to_datetime(merged_data['timestamp'], unit='s')
    
    all_genres = genres['genres'].str.split('|').explode().unique()
    all_genres = [g for g in all_genres if g != '(no genres listed)']
    
    # Store all_genres order for downstream so we know what index is what genre
    # E.g. attach to dataframe attrs or just build the dictionaries
    movie_genre_vectors = {
        row['movieId']: genre_to_binary_vector(row['genres'], all_genres)
        for _, row in genres.iterrows()
    }
    
    merged_data['genre_vector'] = merged_data['movieId'].map(movie_genre_vectors)
    
    # Sort chronologically right off the bat per user so we never forget
    merged_data = merged_data.sort_values(['userId', 'timestamp']).reset_index(drop=True)
    
    return merged_data, all_genres
