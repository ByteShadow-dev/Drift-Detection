import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def genre_to_prob_vector(genre_string, all_genres, epsilon):
    if pd.isna(genre_string) or genre_string == '(no genres listed)':
        movie_genres = []
    else:
        movie_genres = genre_string.split('|')
    
    # Binary count vector
    count_vector = np.array([1.0 if g in movie_genres else 0.0 for g in all_genres])
    
    # Laplace smoothing → then normalize to get probability distribution
    smoothed = count_vector + epsilon
    prob_vector = smoothed / smoothed.sum()
    
    return prob_vector

def load_and_prepare(dataset_path):
    # Load MovieLens ratings and genres
    genres = pd.read_csv(f'{dataset_path}/movies.csv')
    ratings = pd.read_csv(f'{dataset_path}/ratings.csv')
    
    merged_data = ratings.merge(genres[['movieId', 'genres']], on='movieId', how='left') # merged the genre and movies into a single dataset
    
    merged_data['date'] = pd.to_datetime(merged_data['timestamp'], unit='s')
    
    all_genres = genres['genres'].str.split('|').explode().unique()
    all_genres = [g for g in all_genres if g != '(no genres listed)']
    
    epsilon = 0.001
    
    # Step 3: Build a movieId → probability vector lookup dictionary
    movie_genre_vectors = {
        row['movieId']: genre_to_prob_vector(row['genres'], all_genres, epsilon)
        for _, row in genres.iterrows()
    }
    
    # Step 4: Map vectors onto your merged_data
    merged_data['genre_vector'] = merged_data['movieId'].map(movie_genre_vectors)
    
    return merged_data