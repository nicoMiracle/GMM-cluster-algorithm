#nicole nechita rone8293
import os
import pandas as panda
import seaborn
from sklearn.impute import KNNImputer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as matplot
from sklearn.decomposition import PCA
import numpy

#remove the specified field
def clean_column(ratingDataSetPath,column):
    if(os.path.exists(ratingDataSetPath)):
        dataSet = panda.read_csv(ratingDataSetPath)
        if column in dataSet.columns:
            
            dataSet.drop(columns=column,axis=1,inplace=True)
            dataSet.to_csv(ratingDataSetPath, index=False)
            print("Dropped")
    else:
        print("path to dataset does not exist")

#remove duplicates and save
def check_duplicates(dataset_path,dataset_file_name):
    ratings = panda.read_csv(dataset_path)

    duplicate_rows = ratings.duplicated()
    print(f"Total dublicate rows: {duplicate_rows.sum()}")
    if duplicate_rows.any():
        ratings = ratings.drop_duplicates()
    ratings.to_csv(dataset_file_name,index=False)

#stratify the dataset to reduce size, keeping distribution of ratings
def operation_data_sampling(ratingPath, file_name, fraction_value, rand_sta_value):
    if os.path.exists(ratingPath):
        dataset = panda.read_csv(ratingPath)

        dataset['rating_bins'] = panda.cut(dataset['Rating'], bins=[1, 2, 3, 4, 5], include_lowest=True)

        sampled_portion = dataset.groupby('rating_bins', group_keys=False).apply(lambda x: x.sample(frac=fraction_value, random_state=rand_sta_value))

        sampled_portion = sampled_portion.drop(columns=['rating_bins','TimeStamp'])

        sampled_portion.to_csv(file_name, index=False)

    else:
        print("ERROR: file path does not exist")


#remove low ratings below 20 for users and movies
def remove_low_ratings(ratingPath):
    if(os.path.exists(ratingPath)):
        dataset = panda.read_csv(ratingPath)
    
        min_ratings_user = 20
        min_ratings_movie = 20

        #remove values of movies and users below 20, repeat for efficiency
        for _ in range(2):
            
            user_counts = dataset['UserID'].value_counts()
            active_users = user_counts[user_counts >= min_ratings_user].index
            dataset = dataset[dataset['UserID'].isin(active_users)]

            movie_counts = dataset['MovieID'].value_counts()
            popular_movies = movie_counts[movie_counts >= min_ratings_movie].index
            dataset = dataset[dataset['MovieID'].isin(popular_movies)]

        dataset.to_csv(ratingPath, index=False)


#fill matrix with knn values and save the matrix
def fill_matrix(dataset_path,output_file_name,knn_imputer_neighbors):
    dataset = panda.read_csv(dataset_path)
    user_movie_matrix = dataset.pivot_table(
        index="UserID",
        columns="MovieID",
        values="Rating"
    )
    
    user_movie_matrix_np = user_movie_matrix.values
    
    num_actual_ratings = user_movie_matrix.notna().sum().sum()
    
    knn_imputer = KNNImputer(n_neighbors=knn_imputer_neighbors)
    
    user_item_filled = knn_imputer.fit_transform(user_movie_matrix_np)
    
    user_item_filled_df = panda.DataFrame(user_item_filled, index=user_movie_matrix.index, columns=user_movie_matrix.columns)
    
    print(user_item_filled_df.shape)
    print(user_item_filled_df.size)
    
    
    print(f"Number of actual ratings: {num_actual_ratings}")
    any_missing_values_left = user_item_filled_df.isnull().sum().sum()

    print(f"Total missing values after filling: {any_missing_values_left}")

    user_item_filled_df.to_csv(output_file_name)
    print(user_item_filled_df.head())

#check the distribution of a dataset - how many rated 1-2-3-4-5
def check_distribution(dataset_path,fig_name):
        dataset = panda.read_csv(dataset_path)
        matplot.figure(figsize=(7, 5))
        seaborn.histplot(dataset['Rating'], kde=False,discrete=True, bins=20, color="red")
        matplot.title(fig_name)
        matplot.savefig(fig_name, dpi=300)
        matplot.show()

#check matrix for sparsity
def matrix_sparsity(matrix_path):
    matrix = panda.read_csv(matrix_path)
    total = matrix.size
    print(matrix.shape)
    non_zero = (matrix != 0).sum().sum()
    return 100 * (1 - non_zero / total)


#split a dataset into a training and test set
def split_train_test(datasetPath,train_file_name,test_file_name,rand_sta):   
    dataset = panda.read_csv(datasetPath)
    train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=rand_sta)
    train_df.to_csv(train_file_name, index=False)
    test_df.to_csv(test_file_name, index=False)


def test_bic_aic(dataset_path,fig_name,max_itr,rand_sta,pca_components,max_gmm_comp):
    train_dataset = panda.read_csv(dataset_path)

    #fit into PCA
    pca = PCA(n_components=pca_components)  # keep 95% of variance
    
    values = train_dataset.values
    reduced_dim = pca.fit_transform(values)

    print(f"Reduced from {values.shape[1]} to {reduced_dim.shape[1]} dimensions")
    n_values = numpy.arange(1,max_gmm_comp)

    bic = []
    aic= []

    for n in n_values:
        print(f"Fitting GMM with n >")
        gmm = GaussianMixture(n_components=n, covariance_type='full', max_iter=max_itr, random_state=rand_sta)
        gmm.fit(reduced_dim)

        bic.append(gmm.bic(reduced_dim))
        aic.append(gmm.aic(reduced_dim))

    matplot.figure(figsize=(12, 6))
    matplot.plot(n_values, bic, label='BIC', marker='o')
    matplot.plot(n_values, aic, label='AIC', marker='o')
    matplot.xlabel('Number of Clusters')
    matplot.ylabel('Score')
    matplot.title('Model Selection: BIC and AIC for GMM')
    matplot.legend()
    matplot.grid(True)
    matplot.savefig(fig_name)
    matplot.show()

#Step one check for duplicates and remove them
orig_dataset_path = 'datasets/ratings.csv'
dataset_witho_duplicates = "ratings_no_dup.csv"
check_duplicates(orig_dataset_path, dataset_witho_duplicates)
check_distribution(dataset_witho_duplicates,"Full dataset after duplicates removed")

#Step 2 - sample the dataset by 70% to size it down while keeping distribution
dataset_sampled="stratified_dataset.csv"
operation_data_sampling(dataset_witho_duplicates,dataset_sampled,0.7,42)
check_distribution(dataset_sampled,"Dataset distribution after sampling")


#Step 3 -  remove too low ratings
remove_low_ratings(dataset_sampled)
check_distribution(dataset_sampled,"Sampled dataset after low filtering")

#Step 4 - Split into training and test
training_file_name = "training_set.csv"
test_file_name = "test_set.csv"
split_train_test(dataset_sampled,training_file_name,test_file_name,42)

#Step 5 - Create matrix and fill it in with KNN imputer - this will take several minutes
output_matrix_file = "filled_matrix.csv"
fill_matrix(training_file_name,output_matrix_file,6)

#Step 6 - check matrix sparsity
matrix_sparsity(output_matrix_file)

#Step 7 - Test AIC BIC - this will take a bit
output_matrix_file = "filled_matrix.csv"
test_bic_aic(output_matrix_file,"BIC_AIC distribution",500,42,0.95,50)