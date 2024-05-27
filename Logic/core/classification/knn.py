import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

from basic_classifier import BasicClassifier
from data_loader import ReviewLoader
from collections import Counter

class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors
        self.X = None
        self.y = None

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        self
            Returns self as a classifier
        """
        self.X = x
        self.y = y

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        predictions = []
        for i in tqdm(range(x.shape[0])):
            distances = np.linalg.norm(self.X - x[i, :], axis=1)
            nearest_neighbors_indices = np.argsort(distances)[:self.k]
            nearest_neighbors_labels = self.y[nearest_neighbors_indices]
            most_common_label = Counter(nearest_neighbors_labels).most_common(1)[0][0]
            predictions.append(most_common_label)
        return np.array(predictions)


    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        y_pred = self.predict(x)
        return classification_report(y, y_pred)



# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    review_loader = ReviewLoader(file_path='/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/IMDB Dataset.csv')
    review_loader.load_data()
    review_loader.get_embeddings()
    x_train, x_test, y_train, y_test = review_loader.split_data(test_data_ratio=0.3)
    print('KNN classifier starting to train!')
    knn_classifier = KnnClassifier(n_neighbors = 5)
    knn_classifier.fit(x_train, y_train)

    # Predict and print classification report
    report = knn_classifier.prediction_report(x_test, y_test)
    print(report)
