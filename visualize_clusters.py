import matplotlib.pyplot as plt
from clustering import gmm_clustering
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os


def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()

    if covariance.shape == (2, 2):
        U, s, _ = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw 1σ, 2σ, and 3σ ellipses
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position,
                             nsig * width,
                             nsig * height,
                             angle=angle,
                             **kwargs))


def visualize_gmm_clusters(train_set, cluster_labels, gmm, number_of_clusters, save_dir="clustering_images"):
    os.makedirs(save_dir, exist_ok=True)

    # Reduce to 2D with PCA
    pca = PCA(n_components=2)
    X_principal = pd.DataFrame(pca.fit_transform(train_set.values), columns=['P1', 'P2'])
    centers_2d = pca.transform(gmm.means_)

    # Plot points and cluster centers
    fig, ax = plt.subplots(figsize=(8, 6))  # slightly bigger for better detail
    scatter = ax.scatter(X_principal['P1'], X_principal['P2'],
                         c=cluster_labels, cmap='viridis', s=40, alpha=0.7, edgecolors='k', linewidth=0.3)

    ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
               c='red', s=200, marker='*', label='Centers', edgecolors='black', linewidths=1.2)

    # Plot ellipses around each GMM component
    w_factor = 0.2 / gmm.weights_.max()
    for i, (mean, covar, w) in enumerate(zip(gmm.means_, gmm.covariances_, gmm.weights_)):
        mean_2d = pca.transform(mean.reshape(1, -1))[0]
        cov_2d = pca.components_ @ covar @ pca.components_.T
        draw_ellipse(mean_2d, cov_2d, ax=ax,
                     alpha=0.15, facecolor='steelblue', edgecolor='black', linewidth=1.5)

    ax.set_title(f'GMM Clustering Visualization (k={number_of_clusters})')
    ax.set_xlabel('PCA - Component 1')
    ax.set_ylabel('PCA - Component 2')
    ax.grid(True)
    ax.legend()


    filename = f"{save_dir}/gmm_pca_scatter_k{number_of_clusters}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved cluster plot with ellipses to {filename}")

if __name__ == "__main__":
    k=6
    print(f"\nRunning GMM clustering and visualization for k={k}...")
    train_labels, gmm, train_set = gmm_clustering("training_set.csv","test_set.csv",k)
    visualize_gmm_clusters(train_set, train_labels, gmm, k)
    