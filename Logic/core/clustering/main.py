import numpy as np
import os
from sklearn.calibration import LabelEncoder
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import pandas as pd

from dimension_reduction import DimensionReduction
#from clustering_metrics import ClusteringMetrics
from clustering_utils import ClusteringUtils

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from word_embedding.fasttext_data_loader import FastTextDataLoader
from word_embedding.fasttext_model import FastText
from word_embedding.fasttext_data_loader import preprocess_text

"""
# Main Function: Clustering Tasks

# 0. Embedding Extraction
# Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.
"""
file_path = '/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/IMDB_crawled_given.json'
fasttext_loader = FastTextDataLoader(file_path)
X, y = fasttext_loader.create_train_data()
fasttext_model = FastText(method='skipgram')
fasttext_model.load_model('FastText_model.bin')
print('generating embeddings: ...')
embeddings = [fasttext_model.get_query_embedding(summary) for summary in tqdm(X)]

"""
# 1. Dimension Reduction TESTED AND OK
#       Perform Principal Component Analysis (PCA):
#     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
#     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
#     - Draw plots to visualize the results.
"""
dimension_reduction = DimensionReduction()

n_components = 30
print('doing PCA!')
pca_embeddings = dimension_reduction.pca_reduce_dimension(embeddings, n_components=n_components)
project_name="PCA_Project"
run_name="PCA_Run"
dimension_reduction.wandb_plot_explained_variance_by_components(embeddings, project_name=project_name, run_name=run_name)


"""
#       Implement t-SNE (t-Distributed Stochastic Neighbor Embedding): TESTED AND OK
#     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method. 
        TESTED AND OK
#     - Use the output vectors from this step to draw the diagram. TESTED AND OK
"""
# print('doing TSNE!')
# tsne_embeddings = dimension_reduction.convert_to_2d_tsne(pca_embeddings)
# dimension_reduction.wandb_plot_2d_tsne(pca_embeddings, y, project_name="TSNE_Project", run_name="TSNE_Run")

"""
# 2. Clustering
## K-Means Clustering
#       Implement the K-means clustering algorithm from scratch. TESTED AND OK
#       Create document clusters using K-Means.
#       Run the algorithm with several different values of k.
#       For each run:
#     - Determine the genre of each cluster based on the number of documents in each cluster. TESTED AND OK
#     - Draw the resulting clustering using the two-dimensional vectors from the previous section. TESTED AND OK
#     - Check the implementation and efficiency of the algorithm in clustering similar documents. TESTED AND OK
#       Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
#       Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)
"""
clustering_util = ClusteringUtils()
# print('doing K-means on TSNE data')
# clustering_util.visualize_kmeans_clustering_wandb(tsne_embeddings.tolist(), 6, project_name, run_name)
# k_values = [4, 6, 7, 8, 9, 10, 12, 14, 16]
# print('doing K-means')
# clustering_util.plot_kmeans_cluster_scores(pca_embeddings.tolist(), y.tolist(), k_values, project_name, run_name)
"""
## Hierarchical Clustering
#       Perform hierarchical clustering with all different linkage methods. TESTED AND OK
#       Visualize the results. TESTED AND OK

# 3. Evaluation
#       Using clustering metrics, evaluate how well your clustering method is performing. TESTED AND OK
"""
linkage_methods = ['single', 'complete', 'average', 'ward']
for link_method in linkage_methods:
    clustering_util.wandb_plot_hierarchical_clustering_dendrogram(pca_embeddings.tolist(), project_name, linkage_method=link_method, run_name=run_name)