from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# import wandb


class DimensionReduction:

    def __init__(self):
        # wandb.init(project='MIR PHASE 1', entity='kasramlh')
        # not using wandb because the site wont open with/without VPN!!
        self.pca = PCA()
        self.tsne_2d = TSNE(n_components=2, random_state=42)

    def pca_reduce_dimension(self, embeddings, n_components):
        """
        Performs dimensional reduction using PCA with n components left behind.

        Parameters
        ----------
            embeddings (list): A list of embeddings of documents.
            n_components (int): Number of components to keep.

        Returns
        -------
            list: A list of reduced embeddings.
        """
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        return reduced_embeddings

    def convert_to_2d_tsne(self, emb_vecs):
        """
        Converts each raw embedding vector to a 2D vector.

        Parameters
        ----------
            emb_vecs (list): A list of vectors.

        Returns
        --------
            list: A list of 2D vectors.
        """
        tsne_2d = TSNE(n_components=2, random_state=42)
        tsne_embeddings = tsne_2d.fit_transform(emb_vecs)
        return tsne_embeddings

    def wandb_plot_2d_tsne(self, data, y, project_name, run_name):
        """ This function performs t-SNE (t-Distributed Stochastic Neighbor Embedding) dimensionality reduction on the input data and visualizes the resulting 2D embeddings by logging a scatter plot to Weights & Biases (wandb).

        t-SNE is a widely used technique for visualizing high-dimensional data in a lower-dimensional space, typically 2D. It aims to preserve the local structure of the data points while capturing the global structure as well. This function applies t-SNE to the input data and generates a scatter plot of the resulting 2D embeddings, allowing for visual exploration and analysis of the data's structure and potential clusters.

        The scatter plot is a useful way to visualize the t-SNE embeddings, as it shows the spatial distribution of the data points in the reduced 2D space.

        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform t-SNE dimensionality reduction on the input data, obtaining 2D embeddings.
        3. Create a scatter plot of the 2D embeddings using matplotlib.
        4. Log the scatter plot as an image to the wandb run, allowing visualization of the t-SNE embeddings.

        Parameters
        -----------
        data: np.ndarray
            The input data to perform t-SNE dimensionality reduction on.
        project_name: str
            The name of the wandb project to log the t-SNE scatter plot.
        run_name: str
            The name of the wandb run to log the t-SNE scatter plot.

        Returns
        --------
        None
        """
        # Initialize wandb
        # run = wandb.init(project=project_name, name=run_name)

        # Perform t-SNE dimensionality reduction
        tsne_embeddings = self.convert_to_2d_tsne(data)
        # Plot the t-SNE embeddings
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], lw=0.1, c=y, cmap=plt.cm.get_cmap('jet', 10)) # type: ignore
        plt.colorbar(ticks=range(10), label='digit value')
        plt.clim(-0.5, 9.5)
        plt.title("t-SNE 2D Embeddings")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()
        

        # Log the plot to wandb
        # wandb.log({"t-SNE 2D Embeddings": wandb.Image(plt)})

        # Close the plot display window if needed (optional)
        # plt.close()

    def wandb_plot_explained_variance_by_components(self, data, project_name, run_name):
        """
        This function plots the cumulative explained variance ratio against the number of components for a given dataset and logs the plot to Weights & Biases (wandb).

        The cumulative explained variance ratio is a metric used in dimensionality reduction techniques, such as Principal Component Analysis (PCA), to determine the amount of information (variance) retained by the selected number of components. It helps in deciding how many components to keep while balancing the trade-off between retaining valuable information and reducing the dimensionality of the data.

        The function performs the following steps:
        1. Fit a PCA model to the input data and compute the cumulative explained variance ratio.
        2. Create a line plot using Matplotlib, where the x-axis represents the number of components, and the y-axis represents the corresponding cumulative explained variance ratio.
        3. Initialize a new wandb run with the provided project and run names.
        4. Log the plot as an image to the wandb run, allowing visualization of the explained variance by components.


        Parameters
        -----------
        data: np.ndarray
            The input data for which the explained variance by components will be computed and plotted.
        project_name: str
            The name of the wandb project to log the explained variance plot.
        run_name: str
            The name of the wandb run to log the explained variance plot.

        Returns
        --------
        None
        """

        pca = PCA().fit(data)
        cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

        # Create the plot
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
        plt.title("Explained Variance by Number of Components")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance Ratio")
        plt.show()
        # Initialize wandb
        # run = wandb.init(project=project_name, name=run_name)

        # Log the plot to wandb
        # wandb.log({"Explained Variance": wandb.Image(plt)})

        # Close the plot display window if needed (optional)
        # plt.close()
