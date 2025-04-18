#nicole nechita rone8293

import collections
import pandas as panda
import os
import seaborn as sns
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as matplot
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, root_mean_squared_error,precision_score,recall_score
import numpy
from collections import defaultdict
import data_preprocessing_updated as pre_process

DATASET_DIR = "created_datasets"
PLOT_DIR = "plot_distributions"
FILLED_PLOT_DIR = "filled_plot_distributions"
BIC_AIC_DIR = "bic_aic_results"

def predict_rating(user_index, movie_col, user_cluster_prob, cluster_movie_means):

    rating_total = 0
    prob_total = 0

    for cluster_id in range(user_cluster_prob.shape[1]):

        movie_ratings = cluster_movie_means[cluster_id]
        if movie_col in movie_ratings and not numpy.isnan(movie_ratings[movie_col]):

            movie_mean_in_cluster = movie_ratings[movie_col]
            prob = user_cluster_prob[user_index, cluster_id]
            rating_total += prob * movie_mean_in_cluster
            prob_total += prob

    if prob_total > 0:
        return rating_total / prob_total
    else:
        return numpy.nan

def print_terminal_stats(cluster_nr,mae,rmse,precision,recall,user_clusters,precision_at,recall_at):
    print(f"\n Evluation at cluster {cluster_nr}) >")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision@K: {precision_at:.4f}")
    print(f"Recall@K: {recall_at:.4f}")

    cluster_counts = collections.Counter(user_clusters)
    print("Cluster sizes:", dict(cluster_counts))

 # plot metrics into a plot / graph
def plot_metrics(metrics_list,metric_dir, save_prefix):

    os.makedirs(metric_dir,exist_ok=True)

    for metric in metrics_list:
        if metric == "k":
            continue  # skip the first metric, its used for x
        matplot.figure(figsize=(10, 5))
        matplot.plot(metrics_list["k"], metrics_list[metric], marker="o", label=metric, color="steelblue")
        matplot.xlabel("Number of Clusters (k)")
        matplot.ylabel(metric)
        matplot.title(f"{metric} on varying GMM Clusters")
        matplot.grid(True)
        matplot.legend()
        matplot.tight_layout()
        filename = os.path.join(metric_dir,f"{save_prefix}_{metric.lower()}.png")
        matplot.savefig(filename)

#check the distribution of probabilities across users
def plot_confidence (run_nr,test_cluster_prob,cluster_nr):
    os.makedirs(f"confidence_map_run {run_nr}",exist_ok=True)
    max_probs = test_cluster_prob.max(axis=1)
    matplot.clf()
    matplot.hist(max_probs, bins=20, color='mediumpurple', edgecolor='black')
    matplot.title(f"Cluster confidence k = {cluster_nr}")
    matplot.xlabel("Highest Cluster Probability")
    matplot.ylabel("Number of Users")
    filename = os.path.join(f"confidence_map_run {run_nr}",f"confidence at k{cluster_nr}.png")
    matplot.savefig(filename)
    matplot.grid(True)
    matplot.close()

#plot the heatmap for each run at each value k in gmm, to check probability distribution
def plot_heatmap(run_nr,test_cluster_prob,cluster_nr):

    heatmap_dir = f"cluster_prob_heatmaps_run{run_nr}"
    os.makedirs(heatmap_dir, exist_ok=True)

    num_users_to_plot = 3000
    matplot.clf()

    matplot.figure(figsize=(20, 15))
    sns.heatmap(test_cluster_prob[:num_users_to_plot],
                cmap="viridis",
                cbar=True,
                xticklabels=[f"Cluster {i}" for i in range(test_cluster_prob.shape[1])],
                yticklabels=[f"User {i}" for i in range(num_users_to_plot)])

    matplot.title(f"User cluster probability heatmap {num_users_to_plot} Users - k = {cluster_nr}")
    matplot.xlabel("Clusters")
    matplot.ylabel("Users")
    matplot.tight_layout()

    heatmap_path = os.path.join(heatmap_dir, f"cluster_prob_heatmap_k{cluster_nr}.png")
    matplot.savefig(heatmap_path)
    matplot.close()

def run_gmm(train_matrix_path,test_matrix_path,sparse_test_path,run_nr,gmm_rand_state,pca_rand_state,min_n,max_n):
    max_itr = 500
    relevant_trhold = 4.0
    pca_comp = 2

    metrics_per_k = {
        "k": [],
        "MAE": [],
        "RMSE": [],
        "Precision": [],
        "Recall": [],
        "Precision_topN":[],
        "Recall_topN" : []
    }

    for cluster_nr in range(min_n, max_n):
        print(f"\n GMM with cluster ={cluster_nr} >")

        #Load the necessary data
        train_matrix = panda.read_csv(train_matrix_path, index_col=0)
        test_matrix = panda.read_csv(test_matrix_path, index_col=0)

        sparse_test_matrix = panda.read_csv(sparse_test_path, index_col=0)

        # Apply PCA
        pca_transformer = PCA(n_components=pca_comp, random_state=pca_rand_state)
        train_matrix_pca = pca_transformer.fit_transform(train_matrix.values)
        test_matrix_pca = pca_transformer.transform(test_matrix.values)
        print("Fitted matrix into PCA")

        # Fit GMM
        gmm = GaussianMixture(n_components=cluster_nr, covariance_type='full', max_iter=max_itr, random_state=gmm_rand_state)
        gmm.fit(train_matrix_pca)
        print("GMM fitted >")

        # Get soft probability 
        test_cluster_prob = gmm.predict_proba(test_matrix_pca)
        print("Predicted soft prob >")


        #plot confidence map and heatmap of probabilities
        plot_confidence(run_nr,test_cluster_prob,cluster_nr)
        plot_heatmap(run_nr,test_cluster_prob,cluster_nr)
        print("Plots plotted >")
        
        # filter by clusters and find cluster mean
        user_clusters = gmm.predict(train_matrix_pca)
        train_matrix["Cluster"] = user_clusters

        movie_means_per_cluster = {}

        for cluster in range(cluster_nr):
                users_in_cluster = train_matrix[train_matrix["Cluster"] == cluster].drop(columns=["Cluster"])
                movie_means_per_cluster[cluster] = users_in_cluster.mean(skipna=True)
        

        # predict ratings for users, save for use in TOP-N measurement
        user_predictions = defaultdict(dict)

        true_ratings = []
        predicted_ratings = []

        for user_index, user_id in enumerate(sparse_test_matrix.index):
            for movie_id in sparse_test_matrix.columns:
                true_rating = sparse_test_matrix.loc[user_id, movie_id]

                if not numpy.isnan(true_rating):
                    pred_rating = predict_rating(user_index, movie_id, test_cluster_prob, movie_means_per_cluster)

                    if not numpy.isnan(pred_rating):
                        true_ratings.append(true_rating)
                        predicted_ratings.append(pred_rating)

                        # save for later during top_n computation
                        user_predictions[user_id][movie_id] = {
                            "true": true_rating,
                            "pred": pred_rating
                        }
        
        TOP_N = 10 

        precision_at_k_total = 0
        recall_at_k_total = 0
        users_evaluated = 0

        for user_id, rating_dict in user_predictions.items():

            if len(rating_dict) >= TOP_N:
                
                #sort by predicted ratings highest
                sorted_movies = sorted(rating_dict.items(), key=lambda x: x[1]["pred"], reverse=True)
                top_N_movies = sorted_movies[:TOP_N]
                top_N_movie_ids = [movie_id for movie_id, _ in top_N_movies]

                #put in the movies that pass the relevant treshold
                relevant_movies = sum(1 for movie_id in top_N_movie_ids if rating_dict[movie_id]["true"] >= relevant_trhold)
                
                #how many movies, beyond top N, are relevant?
                total_relevant = sum(1 for movie_data in rating_dict.values() if movie_data["true"] >= relevant_trhold)

                #calculate precision for top N for this user
                precision_at_k_total += relevant_movies / TOP_N
                #calculate recall for this particular user
                recall_at_k_total += relevant_movies / total_relevant if total_relevant > 0 else 0
                users_evaluated += 1

        # find the exact precision and recall at k
        if users_evaluated > 0:
            precision_at_k = precision_at_k_total / users_evaluated
            recall_at_k = recall_at_k_total / users_evaluated
        else:
            precision_at_k = 0
            recall_at_k = 0

        # calculate metrix
        mae = mean_absolute_error(y_true=true_ratings, y_pred= predicted_ratings)
        rmse = root_mean_squared_error(y_true= true_ratings,y_pred= predicted_ratings)

        # place values in their respective bins for precision and recall
        #1 if relevant, 0 if not relevant
        true_bin = [1 if rating >= relevant_trhold else 0 for rating in true_ratings]
        pred_bin = [1 if rating >= relevant_trhold else 0 for rating in predicted_ratings]

        #calculate precision and recal for the entire test
        precision = precision_score(true_bin, pred_bin, zero_division=0)
        recall = recall_score(true_bin, pred_bin, zero_division=0)

        # save metrics
        metrics_per_k["k"].append(cluster_nr)
        metrics_per_k["MAE"].append(mae)
        metrics_per_k["RMSE"].append(rmse)
        metrics_per_k["Precision"].append(precision)
        metrics_per_k["Recall"].append(recall)
        metrics_per_k["Precision_topN"].append(precision_at_k)
        metrics_per_k["Recall_topN"].append(recall_at_k)

        # print data for this cluster k in the terminal
        print_terminal_stats(cluster_nr,mae,rmse,precision,recall,user_clusters,precision_at_k,recall_at_k)

        
    #plot the metrics into pictures
    plot_metrics(metrics_per_k,f"metrics_run_{run_nr}",f"test_results_run_{run_nr}")
    #create CSV file with the numbers for later use
    metrics_df = panda.DataFrame(metrics_per_k)
    metrics_df.to_csv(f"metrics_run_{run_nr}.csv", index=False)

def join_path(directory,file):
    return os.path.join(directory,file)

#MAIN#

#constant for train/test ration, for consistency
TRAIN_TEST_SPLIT = 0.2

#neighbours value will be set to 5 for consistency
KNN_NEIGH = 8

#keep gmm iterations at 500 for consistency
GMM_ITER = 500

#pca component set at 2 for consistency
PCA_COMP = 2

#MAKE DATASETS FOR RUN 
def create_ds_run_gm(run_nr,sample_rand_sta,split_rand_sta,gmm_rand_sta,pca_rand_sta, max_clusters):
    filename_dup_rm = f"ratings_dup_removed_run_{run_nr}"
    pre_process.check_duplicates("datasets/ratings.csv",f"{filename_dup_rm}")
    pre_process.plot_distribution_dataset(f"{join_path(DATASET_DIR,filename_dup_rm)}.csv",f"data_set_dup_rm_distribution_run_{run_nr}")

    sampled_file_name = f"sampled_dataset_run_{run_nr}"
    pre_process.operation_data_sampling(f"{join_path(DATASET_DIR,filename_dup_rm)}.csv",sampled_file_name,0.7,sample_rand_sta)
    pre_process.plot_distribution_dataset(f"{join_path(DATASET_DIR,sampled_file_name)}.csv",f"data_set_sampled_distribution_run_{run_nr}")

    ##remove low ratings
    pre_process.remove_low_ratings(f"{join_path(DATASET_DIR,sampled_file_name)}.csv",20,20)

    #split into train/test
    train_file = f"train_ds_{run_nr}"
    test_file = f"test_ds_{run_nr}"
    pre_process.split_train_test(f"{join_path(DATASET_DIR,sampled_file_name)}.csv",train_file,test_file,split_rand_sta,TRAIN_TEST_SPLIT)

    #create sparse test matrix for prediction
    sparse_test_mat = f"sparse_tes_mat_{run_nr}"
    pre_process.create_sparse_matrix(f"{join_path(DATASET_DIR,test_file)}.csv",sparse_test_mat)

    #create matrixes, fill with knn imputation, save
    #do not run this if imputed matrixes exist already, this is expensive
    knn_imputed_file = f"run_{run_nr}_knn_imputed"
    train_file_path = join_path(DATASET_DIR,f"{train_file}.csv")

    test_file_path = join_path(DATASET_DIR,f"{test_file}.csv")

    pre_process.fill_matrix_knn(train_file_path,test_file_path,
                                knn_imputed_file,KNN_NEIGH)


    filled_train_mat = join_path(DATASET_DIR,f"{knn_imputed_file}_train.csv")
    filled_test_mat = join_path(DATASET_DIR,f"{knn_imputed_file}_test.csv")
    sparse_test_mat_path = join_path(DATASET_DIR,f"{sparse_test_mat}.csv")

    pre_process.test_bic_aic(filled_train_mat,
                            "bic_aic_results",GMM_ITER,gmm_rand_sta,pca_rand_sta,PCA_COMP,max_clusters,run_nr)

    run_gmm(filled_train_mat,filled_test_mat,sparse_test_mat_path,run_nr,gmm_rand_sta,pca_rand_sta,2,max_clusters)


# run 1
create_ds_run_gm(1,2,10,20,15,51)
#run 2
create_ds_run_gm(2,13,40,42,25,51)
#run 3
create_ds_run_gm(3,12,6,31,3,51)
