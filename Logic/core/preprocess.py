from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import re
import string

class Preprocessor:

    def __init__(self, documents: list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        self.documents = documents
        self.stopwords = ['this', 'that', 'about', 'whom', 'being', 'where', 'why', 'had', 'should', 'each']
        self.stemmer = PorterStemmer()

    def preprocessQuery(self, query):
        #TODO
        return self.normalize(query)

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
        # return a list of dicts!
        """
        each doc is in form of:

            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]

        """
        preprocessed_documents = []
        for doc in self.documents:
            doc['title'] = self.normalize(doc['title'])
            doc['first_page_summary'] = self.normalize(doc['first_page_summary'])
            doc['directors'] = self.normalize(doc['directors'])
            doc['writers'] = self.normalize(doc['writers'])
            doc['stars'] = self.normalize(doc['stars'])
            doc['genres'] = self.normalize(doc['genres'])
            doc['languages'] = self.normalize(doc['languages'])
            doc['countries_of_origin'] = self.normalize(doc['countries_of_origin'])
            doc['summaries'] = self.normalize(doc['summaries'])
            doc['synopsis'] = self.normalize(doc['synopsis'])
            doc['reviews'] = self.normalize(flatten(doc['reviews']))
            preprocessed_documents.append(doc)
        return preprocessed_documents

    def normalize(self, text):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        
        if isinstance(text, list):
            res = []
            for temp in text:
                lower_case_text = temp.lower()
                no_link_text = self.remove_links(lower_case_text)
                no_punc_text = self.remove_punctuations(no_link_text)
                no_stop_word = self.remove_stopwords(no_punc_text)
                tokens = self.tokenize(' '.join(no_stop_word))
                stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
                res.append(stemmed_tokens)
            return flatten(res)
        if isinstance(text, str):
            lower_case_text = text.lower()
            no_link_text = self.remove_links(lower_case_text)
            no_punc_text = self.remove_punctuations(no_link_text)
            no_stop_word = self.remove_stopwords(no_punc_text)
            tokens = self.tokenize(' '.join(no_stop_word))
            stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
            return ' '.join(stemmed_tokens)

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        return word_tokenize(text)

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return filtered_words
    
def flatten(xss):
    if xss is None:
        return None
    return [x for xs in xss for x in xs]