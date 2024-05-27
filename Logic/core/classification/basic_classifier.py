import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from word_embedding.fasttext_model import FastText


class BasicClassifier:
    def __init__(self):
        # raise NotImplementedError()
        pass

    def fit(self, x, y):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def prediction_report(self, x, y):
        raise NotImplementedError()

    def get_percent_of_positive_reviews(self, sentences):
        """
        Get the percentage of positive reviews in the given sentences
        Parameters
        ----------
        sentences: list
            The list of sentences to get the percentage of positive reviews
        Returns
        -------
        float
            The percentage of positive reviews
        """

        """
        if not hasattr(self, 'model'):
            raise ValueError("Model is not trained or loaded")
        """

        positive_count = 0
        total_count = len(sentences)
        fasttext_model = FastText(method='skipgram')
        fasttext_model.load_model('FastText_model.bin')
        for sentence in tqdm(sentences):
            prediction = self.predict([fasttext_model.get_query_embedding(sentence)])[0]
            if prediction == 1:
                positive_count += 1

        return (positive_count / total_count) * 100

