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
from visualize_clusters import run_gmm_visualization

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

def print_terminal_stats(cluster_nr,mae,rmse,precision,recall,user_clusters):
    print(f"\n Evluation at cluster {cluster_nr}) >")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

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
    relevant_trhold = 3.0
    pca_comp = 0.1

    metrics_per_k = {
        "k": [],
        "MAE": [],
        "RMSE": [],
        "Precision": [],
        "Recall": []
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

        # Get hard clustering and soft probability 
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
        

        # predict ratings for users
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

        # calculate metrix
        mae = mean_absolute_error(y_true=true_ratings, y_pred= predicted_ratings)
        rmse = root_mean_squared_error(y_true= true_ratings,y_pred= predicted_ratings)

        # place values in their respective bins for precision and recall
        #1 if relevant, 0 if not relevant
        true_bin = [1 if rating >= relevant_trhold else 0 for rating in true_ratings]
        pred_bin = [1 if rating >= relevant_trhold else 0 for rating in predicted_ratings]

        precision = precision_score(true_bin, pred_bin, zero_division=0)
        recall = recall_score(true_bin, pred_bin, zero_division=0)

        # save metrics
        metrics_per_k["k"].append(cluster_nr)
        metrics_per_k["MAE"].append(mae)
        metrics_per_k["RMSE"].append(rmse)
        metrics_per_k["Precision"].append(precision)
        metrics_per_k["Recall"].append(recall)

        # print data for this cluster k in the terminal
        print_terminal_stats(cluster_nr,mae,rmse,precision,recall,user_clusters)

        
    #plot the metrics into pictures
    plot_metrics(metrics_per_k,f"metrics_run_{run_nr}",f"test_results_run_{run_nr}")
    #create CSV file with the numbers for later use
    metrics_df = panda.DataFrame(metrics_per_k)
    metrics_df.to_csv(f"metrics_run_{run_nr}", index=False)


#MAIN#

run_gmm("created_datasets/mean_imputed_train.csv","created_datasets/mean_imputed_test.csv",
        "created_datasets\sparse_test_matrix.csv",1,42,42,10,15)