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
doc = "Over the course of several years, two convicts form a friendship, seeking consolation and, eventually, redemption through basic compassion.Chronicles the experiences of a formerly successful banker as a prisoner in the gloomy jailhouse of Shawshank after being found guilty of a crime he did not commit. The film portrays the man's unique way of dealing with his new, torturous life; along the way he befriends a number of fellow prisoners, most notably a wise long-term inmate named Red.\u2014J-S-Golden When an innocent male banker is sent to prison accused of murdering his wife, he does everything that he can over the years to break free and escape from prison. While on the inside, he develops a friendship with a fellow inmate that could last for years.\u2014RECB3 After the murder of his wife, hotshot banker Andrew Dufresne is sent to Shawshank Prison, where the usual unpleasantness occurs. Over the years, he retains hope and eventually gains the respect of his fellow inmates, especially longtime convict \"Red\" Redding, a black marketeer, and becomes influential within the prison. Eventually, Andrew achieves his ends on his own terms.\u2014Reid Gagle Andy Dufresne is sent to Shawshank Prison for the murder of his wife and her secret lover. He is very isolated and lonely at first, but realizes there is something deep inside your body that people can't touch or get to....'HOPE'. Andy becomes friends with prison 'fixer' Red, and Andy epitomizes why it is crucial to have dreams. His spirit and determination lead us into a world full of imagination, one filled with courage and desire. Will Andy ever realize his dreams?\u2014Andy Haque"
query = "Redemption best movie"
snippet, not_exist_words = snippet_finder.find_snippet(doc, query)
print("Snippet:", snippet)
print("Words not found in the document:", not_exist_words)