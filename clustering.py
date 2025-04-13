from data_preprocessing import preprocess_data
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg') # Important for plot windows in VS Code
import matplotlib.pyplot as plt
import pandas as pd


def gmm_clustering(train_ds,test_ds,number_of_clusters):
  
  #Load preprocessed and split data
  train_set = pd.read_csv(train_ds, index_col=0)
  test_set = pd.read_csv(test_ds, index_col=0)  
 
  #Apply GMM to training set only
  gmm = GaussianMixture(n_components=number_of_clusters, covariance_type='full',  random_state=42)
  gmm.fit(train_set.values)
  

  #Predict clusters
  train_labels = gmm.predict(train_set.values)
  test_labels = gmm.predict(test_set.values)

  

  #Attach cluster Labels to users
  train_user_clusters = pd.DataFrame({
    'UserID': train_set.index,
    'Cluster': train_labels,
  })

  

  test_user_clusters = pd.DataFrame({
    'UserID': test_set.index,
    'Cluster': test_labels,
  })

  return train_labels, gmm, train_set


