import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from basic_classifier import BasicClassifier
from data_loader import ReviewLoader


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

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
        self.classes, counts = np.unique(y, return_counts=True)
        self.num_classes = len(self.classes)
        self.number_of_samples, self.number_of_features = x.shape
        
        self.prior = np.zeros(self.num_classes)
        self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))
        
        for idx, cls in enumerate(self.classes):
            x_cls = x[y == cls]
            self.prior[idx] = x_cls.shape[0] / self.number_of_samples
            self.feature_probabilities[idx] = (x_cls.sum(axis=0) + self.alpha) / (x_cls.sum() + self.alpha * self.number_of_features)
        
        self.log_probs = np.log(self.feature_probabilities)
        self.log_prior = np.log(self.prior)
        
        return self

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
        log_likelihood = x @ self.log_probs.T
        log_posterior = log_likelihood + self.log_prior
        return self.classes[np.argmax(log_posterior, axis=1)]

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
        predictions = self.predict(x)
        return classification_report(y, predictions)

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        x = self.cv.transform(sentences).toarray()
        predictions = self.predict(x)
        positive_reviews = np.sum(predictions == 1)
        return positive_reviews / len(sentences)


# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the revies using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    review_loader = ReviewLoader('/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/IMDB Dataset.csv')
    review_loader.load_data()
    cv = CountVectorizer()
    X = cv.fit_transform(review_loader.review_tokens)
    y = review_loader.sentiments
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    nb = NaiveBayes(count_vectorizer=cv)
    nb.fit(X_train, y_train)
    report = nb.prediction_report(X_test, y_test)
    print(report)