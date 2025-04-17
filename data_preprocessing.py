import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn . model_selection import train_test_split


def preprocess_data():
    
    # Load ratings
    ratings = pd.read_csv('datasets/ratings.csv')

    # Step 1: Remove exact duplicate rows
    duplicate_rows = ratings.duplicated()
    print(f"Total dublicate rows: {duplicate_rows.sum()}")
    if duplicate_rows.any():
        ratings = ratings.drop_duplicates()

    # Step 2: Handle duplicate user-movie ratings by averaging them
    duplicate_user_movie = ratings.duplicated(subset=['UserID', 'MovieID'])
    print(f"Total dublicate user movies: {duplicate_user_movie.sum()}")
    if duplicate_user_movie.any():
        ratings = ratings.groupby(['UserID', 'MovieID'], as_index=False)['Rating'].mean()
    
    num_users = ratings['UserID'].nunique()
    num_movies = ratings['MovieID'].nunique()
    print(f"Unfiltered dataset contains {num_users} users and {num_movies} movies.")

    # Step 3: Filter active users and popular movies
    user_counts = ratings['UserID'].value_counts()
    selected_users = user_counts[user_counts >= 100].index
    ratings = ratings[ratings['UserID'].isin(selected_users)]

    movie_counts = ratings['MovieID'].value_counts()
    selected_movies = movie_counts[movie_counts >= 50].index
    ratings = ratings[ratings['MovieID'].isin(selected_movies)]

    num_users = ratings['UserID'].nunique()
    num_movies = ratings['MovieID'].nunique()
    print(f"Filtered dataset contains {num_users} users and {num_movies} movies.")

    # Step 4: Create User-item Matrix
    user_item_matrix = ratings.pivot_table(index='UserID', columns='MovieID', values='Rating')

    # Step 5: Check for missing-values and if any, fill missing values with user mean
    total_missing_values = user_item_matrix.isnull().sum().sum()
    if total_missing_values > 0:
        user_item_filled = user_item_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)
    else:
        user_item_filled = user_item_matrix
    
    # Confirm all missing values are filled
    any_missing_values_left = user_item_filled.isnull().sum().sum()
    print(f"Total missing values after filling: {any_missing_values_left}")

    # Step 6: Split dataset into train and test test_size 80/20)
    train_set, test_set = train_test_split(user_item_filled, test_size=0.2, random_state=0)
    
  
    train_set.to_csv("datasets/processed_train_dataset.csv")
    test_set.to_csv("datasets/processed_test_dataset.csv")
    user_item_filled.to_csv("datasets/processed_user_item_matrix.csv")

    
if __name__ == "__main__":
    preprocess_data()
