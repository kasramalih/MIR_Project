import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from basic_classifier import BasicClassifier
from data_loader import ReviewLoader


class SVMClassifier(BasicClassifier):
    def __init__(self):
        super().__init__()
        self.model = SVC(C = 5)

    def fit(self, X, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc
        """
        self.model.fit(X, y)

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
        return self.model.predict(x)

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


# F1 accuracy : 78%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    review_loader = ReviewLoader(file_path='/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/IMDB Dataset.csv')
    review_loader.load_data()
    review_loader.get_embeddings()
    x_train, x_test, y_train, y_test = review_loader.split_data(test_data_ratio=0.3)
    #shall we set SVM params? like C, lambda, ...
    svm_classifier = SVMClassifier()
    print('SVM classifier starting to train!')
    svm_classifier.fit(x_train, y_train)

    # Predict and print classification report
    report = svm_classifier.prediction_report(x_test, y_test)
    print(report)

    # Calculate and print the percentage of positive reviews
    sentences = ["I love this movie!", "It was a terrible experience.", "Best film ever!", "Not worth the time."]
    percent_positive = svm_classifier.get_percent_of_positive_reviews(sentences)
    print(f"Percentage of positive reviews: {percent_positive}%")