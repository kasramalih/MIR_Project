class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """
        stopwords = ['this', 'that', 'about', 'whom', 'being', 'where', 'why', 'had', 'should', 'each']
        words = query.split()
        filtered_words = [word for word in words if word not in stopwords]

        return filtered_words

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""
        not_exist_words = []

        doc_tokens = doc.split()
        query_tokens = self.remove_stop_words_from_query(query)

        for query_word in query_tokens:
            # Check if query word exists in the document
            if query_word.lower() in map(str.lower, doc_tokens):  # Case-insensitive comparison
                # Find indices of occurrences of the query word in the document
                indices = [i for i, word in enumerate(doc_tokens) if word.lower() == query_word.lower()]  # Case-insensitive comparison

                for index in indices:
                    # Extract snippet around the query word
                    start_index = max(0, index - self.number_of_words_on_each_side)
                    end_index = min(len(doc_tokens), index + self.number_of_words_on_each_side + 1)
                    snippet_words = doc_tokens[start_index:end_index]

                    # Highlight query word
                    snippet = ' '.join(['***' + word + '***' if word.lower() == query_word.lower() else word for word in snippet_words])

                    final_snippet += snippet + " ... "

            else:
                # Query word not found in document
                not_exist_words.append(query_word)

        final_snippet = final_snippet.strip()  # Remove trailing whitespace

        return final_snippet, not_exist_words

# TODO what if a word is repeated in doc or query? just the first time? shall query be stemmed or not?
    # since the indexed docs are stemmed!
snippet_finder = Snippet()
doc = "The Shawshank Redemption is a movie directed by Frank Darabont."
query = "Redemption best movie"
snippet, not_exist_words = snippet_finder.find_snippet(doc, query)
print("Snippet:", snippet)
print("Words not found in the document:", not_exist_words)