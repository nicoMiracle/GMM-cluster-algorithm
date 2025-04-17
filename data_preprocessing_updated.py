#nicole nechita rone8293
import os
import pandas as panda
import seaborn
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as matplot
from sklearn.decomposition import PCA
import numpy

DATASET_DIR = "created_datasets"
PLOT_DIR = "plot_distributions"
FILLED_PLOT_DIR = "filled_plot_distributions"
BIC_AIC_DIR = "bic_aic_results"

#remove duplicates and save
def check_duplicates(dataset_path,dataset_file_name):
    os.makedirs(DATASET_DIR,exist_ok=True)

    ratings = panda.read_csv(dataset_path)

    duplicate_rows = ratings.duplicated()
    print(f"Total dublicate rows: {duplicate_rows.sum()}")
    if duplicate_rows.any():
        ratings = ratings.drop_duplicates()
    ratings.to_csv(os.path.join(DATASET_DIR,f"{dataset_file_name}.csv"),index=False)

#stratify the dataset to reduce size, keeping distribution of ratings
def operation_data_sampling(dataset_path, file_name, fraction_value, rand_sta_value):

    dataset = panda.read_csv(dataset_path)

    dataset['rating_bins'] = panda.cut(dataset['Rating'], bins=[1, 2, 3, 4, 5], include_lowest=True)

    sampled_portion = dataset.groupby('rating_bins', group_keys=False).apply(lambda x: x.sample(frac=fraction_value, random_state=rand_sta_value))

    sampled_portion = sampled_portion.drop(columns=['rating_bins','TimeStamp'])

    os.makedirs(DATASET_DIR,exist_ok=True)
    sampled_portion.to_csv(os.path.join(DATASET_DIR,f"{file_name}.csv"), index=False)

#remove low ratings below 20 for users and movies
def remove_low_ratings(dataset_path, min_ratings_user,min_ratings_movie):
    dataset = panda.read_csv(dataset_path)

    #remove users given a certain min_value
    user_counts = dataset['UserID'].value_counts()
    active_users = user_counts[user_counts >= min_ratings_user].index
    dataset = dataset[dataset['UserID'].isin(active_users)]

    movie_counts = dataset['MovieID'].value_counts()
    popular_movies = movie_counts[movie_counts >= min_ratings_movie].index
    dataset = dataset[dataset['MovieID'].isin(popular_movies)]

    dataset.to_csv(dataset_path, index=False)


#fill matrix with knn values and save the matrix
def fill_matrix_knn(train_dataset_path, test_dataset_path, output_file_name, knn_imputer_neighbors):
    #load the datasets
    train_dataset = panda.read_csv(train_dataset_path)
    test_dataset = panda.read_csv(test_dataset_path)
    
    #create the train matrix for filling
    train_matrix_Nfilled = train_dataset.pivot_table(
        index="UserID",
        columns="MovieID",
        values="Rating"
    )
    
    #create the test matrix for filling
    test_matrix_Nfilled = test_dataset.pivot_table(
        index="UserID",
        columns="MovieID",
        values="Rating"
    )
    
    knn_imputer = KNNImputer(n_neighbors=knn_imputer_neighbors)

    # fit the imputer on the training data
    knn_imputer.fit(train_matrix_Nfilled)
    
    # fill train and test matrixes with fitted imputer
    train_matrix_fill = knn_imputer.transform(train_matrix_Nfilled)
    test_matrix_fill = knn_imputer.transform(test_matrix_Nfilled)

    # transform to dataframe to save as matrix
    train_user_item_filled_df = panda.DataFrame(train_matrix_fill, index=train_matrix_Nfilled.index, columns=train_matrix_Nfilled.columns)
    test_user_item_filled_df = panda.DataFrame(test_matrix_fill, index=test_matrix_Nfilled.index, columns=test_matrix_Nfilled.columns)

    #check the results
    print(f"count train ratings pre-fill: {train_matrix_Nfilled.notna().sum().sum()}")
    print(f"count test ratings pre-fill: {test_matrix_Nfilled.notna().sum().sum()}")
    
    #check for amount of missing values post-fill
    miss_train_count = train_user_item_filled_df.isnull().sum().sum()
    miss_test_count = test_user_item_filled_df.isnull().sum().sum()
    print(f"missing values in train matrix post-fill: {miss_train_count}")
    print(f"missing values in test matrix post-fill: {miss_test_count}")

    #save to csv file, use index_col=0 when reading later
    os.makedirs(DATASET_DIR,exist_ok=True)
    train_user_item_filled_df.to_csv(os.path.join(DATASET_DIR,f"{output_file_name}_train.csv"))
    test_user_item_filled_df.to_csv(os.path.join(DATASET_DIR,f"{output_file_name}_test.csv"))

def fill_matrix_mean(train_dataset_path, test_dataset_path, output_file_name):
    #load the datasets
    train_dataset = panda.read_csv(train_dataset_path)
    test_dataset = panda.read_csv(test_dataset_path)
    
    #create the train matrix for filling
    train_matrix_Nfilled = train_dataset.pivot_table(
        index="UserID",
        columns="MovieID",
        values="Rating"
    )
    
    #create the test matrix for filling
    test_matrix_Nfilled = test_dataset.pivot_table(
        index="UserID",
        columns="MovieID",
        values="Rating"
    )
    
    simp_imputer = SimpleImputer(strategy="mean")

    #preserve previous ratings by copy
    train_matrix_fill = train_matrix_Nfilled.copy() 
    test_matrix_fill = test_matrix_Nfilled.copy()  
    
    #fit the imputer
    simp_imputer.fit(train_matrix_Nfilled)
    
    # apply the imputer, not overwritting previous ratings
    train_matrix_fill[train_matrix_Nfilled.isna()] = simp_imputer.transform(train_matrix_Nfilled)
    test_matrix_fill[test_matrix_Nfilled.isna()] = simp_imputer.transform(test_matrix_Nfilled)

    # transform to dataframe to save as matrix
    train_user_item_filled_df = panda.DataFrame(train_matrix_fill, index=train_matrix_Nfilled.index, columns=train_matrix_Nfilled.columns)
    test_user_item_filled_df = panda.DataFrame(test_matrix_fill, index=test_matrix_Nfilled.index, columns=test_matrix_Nfilled.columns)

    #check the results
    print(f"count train ratings pre-fill: {train_matrix_Nfilled.notna().sum().sum()}")
    print(f"count test ratings pre-fill: {test_matrix_Nfilled.notna().sum().sum()}")
    
    #check for amount of missing values post-fill
    miss_train_count = train_user_item_filled_df.isnull().sum().sum()
    miss_test_count = test_user_item_filled_df.isnull().sum().sum()
    print(f"missing values in train matrix post-fill: {miss_train_count}")
    print(f"missing values in test matrix post-fill: {miss_test_count}")

    #save to csv file, use index_col=0 when reading later
    os.makedirs(DATASET_DIR,exist_ok=True)
    train_user_item_filled_df.to_csv(os.path.join(DATASET_DIR,f"{output_file_name}_train.csv"))
    test_user_item_filled_df.to_csv(os.path.join(DATASET_DIR,f"{output_file_name}_test.csv"))

#check the distribution of a dataset - how many rated 1-2-3-4-5
def plot_distribution_dataset(dataset_path, fig_name):
    dataset = panda.read_csv(dataset_path)
    
    #create directory if not existing
    os.makedirs(PLOT_DIR, exist_ok=True)
    matplot.clf()
    matplot.figure(figsize=(7, 5))
    seaborn.histplot(dataset['Rating'], kde=False, discrete=True, bins=20, color="red")
    matplot.title(fig_name)
    matplot.savefig(os.path.join(PLOT_DIR,f"{fig_name}.png"), dpi=300)
    matplot.close()

def plot_filled_rating_distribution(filled_matrix_path, filename):
    
    filled_df = panda.read_csv(filled_matrix_path, index_col=0)
    filled_ratings = filled_df.values.flatten()
    filled_ratings = filled_ratings[~numpy.isnan(filled_ratings)]

    # Create save directory if needed
    os.makedirs(FILLED_PLOT_DIR, exist_ok=True)

    # Plot
    matplot.clf()
    matplot.figure(figsize=[12,8])
    matplot.hist(filled_ratings, bins=30, color='cornflowerblue', edgecolor='black')
    matplot.title("Distribution of Filled Ratings")
    matplot.xlabel("Rating")
    matplot.ylabel("Frequency")
    matplot.grid(True)
    matplot.tight_layout()

    save_path = os.path.join(FILLED_PLOT_DIR, f"{filename}.png")
    matplot.savefig(save_path)
    matplot.close()


#check matrix for sparsity
def matrix_sparsity(matrix_path):
    matrix = panda.read_csv(matrix_path,index_col=0)

    total = matrix.size
    print(matrix.shape)
    print(matrix.head(10))
    non_zero = (matrix != 0).sum().sum()
    return 100 * (1 - non_zero / total)


#split a dataset into a training and test set
def split_train_test(datasetPath,train_file_name,test_file_name,rand_sta,test_sz):
    os.makedirs(DATASET_DIR,exist_ok=True)

    dataset = panda.read_csv(datasetPath)
    train_df, test_df = train_test_split(dataset, test_size=test_sz, random_state=rand_sta)

    #save the train and test datasets
    train_df.to_csv(os.path.join(DATASET_DIR,train_file_name), index=False)
    test_df.to_csv(os.path.join(DATASET_DIR,test_file_name), index=False)


def test_bic_aic(dataset_path,fig_name,max_itr,rand_sta,pca_components,max_gmm_comp,run_nr):
    train_dataset = panda.read_csv(dataset_path,index_col=0)

    #fit into PCA
    pca = PCA(n_components=pca_components,random_state=rand_sta)  # keep 95% of variance
    
    values = train_dataset.values
    transformed_matrix = pca.fit_transform(values)

    clust_numbers = numpy.arange(1,max_gmm_comp)

    bic = []
    aic= []

    for n in clust_numbers:
        print(f"Fitting GMM with n >")
        gmm = GaussianMixture(n_components=n, covariance_type='full', max_iter=max_itr, random_state=rand_sta)
        gmm.fit(transformed_matrix)

        bic.append(gmm.bic(transformed_matrix))
        aic.append(gmm.aic(transformed_matrix))

    os.makedirs(BIC_AIC_DIR, exist_ok=True)

    matplot.clf()
    matplot.figure(figsize=(12, 6))
    matplot.plot(clust_numbers, bic, label='BIC', marker='o')
    matplot.plot(clust_numbers, aic, label='AIC', marker='o')
    matplot.xlabel('Number of Clusters')
    matplot.ylabel('Score')
    matplot.title('BIC and AIC results for given dataset')
    matplot.legend()
    matplot.grid(True)
    fig_path = os.path.join(BIC_AIC_DIR,f"{fig_name}_{run_nr}.png")
    matplot.savefig(fig_path)
    matplot.close()


fill_matrix_mean(f"{DATASET_DIR}/training_set.csv",f"{DATASET_DIR}/test_set.csv","mean_imputed")

plot_filled_rating_distribution(f"{DATASET_DIR}/mean_imputed_train.csv","mean_training_filled")
plot_filled_rating_distribution(f"{DATASET_DIR}/mean_imputed_test.csv","mean_test_filled")