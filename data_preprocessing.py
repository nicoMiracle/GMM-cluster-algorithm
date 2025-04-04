import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def preprocess_data():
    # Load ratings
    ratings = pd.read_csv('datasets/ratings.csv')

    # Step 1: Remove exact duplicate rows
    duplicate_rows = ratings.duplicated()
    #print(f"Total dublicate rows: {duplicate_rows}")
    if duplicate_rows.any():
        ratings = ratings.drop_duplicates()

    # Step 2: Handle duplicate user-movie ratings by averaging them
    duplicate_user_movie = ratings.duplicated(subset=['UserID', 'MovieID'])
    #print(f"Total dublicate user movies: {duplicate_user_movie}")
    if duplicate_user_movie.any():
        ratings = ratings.groupby(['UserID', 'MovieID'], as_index=False)['Rating'].mean()

    # Step 3: Filter out movies with fewer than 20 ratings
    movie_counts = ratings['MovieID'].value_counts()
    valid_movies = movie_counts[movie_counts >= 20].index
    ratings = ratings[ratings['MovieID'].isin(valid_movies)]

    # Step 4: Create User-item Rating Matrix
    user_item_matrix = ratings.pivot_table(index='UserID', columns='MovieID', values='Rating')

    # Step 5: Check for missing-values and if any, fill missing values with user mean
    total_missing_values = user_item_matrix.isnull().sum().sum()
    if total_missing_values > 0:
        user_item_filled = user_item_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)
    else:
        user_item_filled = user_item_matrix
    
    # Confirm all missing values are filled
    #any_missing_values_left = user_item_filled.isnull().sum().sum()
    #print(f"Total missing values after filling: {any_missing_values_left}")

    # Step 6: Apply PCA
    pca = PCA(n_components=0.95)
    reduced_user_item_matrix = pca.fit_transform(user_item_filled)

    #print(f"Original shape: {user_item_normalized.shape}")
    #print(f"Reduced shape: {user_item_reduced.shape}")
    #print(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.2%}")

    # Returns processed matrix, rating and orginal matrix
    return reduced_user_item_matrix, ratings, user_item_matrix
