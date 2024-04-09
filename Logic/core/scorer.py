import json
import math
import numpy as np

class Scorer:    
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents

    def get_list_of_documents(self,query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.
        
        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.
            
        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))
    
    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.
        
        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        df = self.idf.get(term, None)
        if df is None:
            df = len(self.index.get(term, {}))
            # print(df)
            # idf = math.log(self.N / (df + 1))  # Adding 1 to avoid division by zero
            self.idf[term] = df
        return df
    
    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """
        
        query_tfs = {}
        if isinstance(query, list):
            for term in query:
                query_tfs[term] = query_tfs.get(term, 0) + 1
        else:
            for term in query.split():
                query_tfs[term] = query_tfs.get(term, 0) + 1

        return query_tfs


    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)): e.x.: lnc.ltn
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        """
        n: no idf ... 
        l: logarithmic tf ( 1 + log(tf_t,d) )
        t: idf (t in second column) ( log(N/df) )
        c: cosine normalization ... 
        """
        # we have tf_t,d
        # we have tf_t,q
        # we have idf
        docs_to_scores = {}
        docs = self.get_list_of_documents(query)
        method_list = [char for char in method if char != '.']
        tf_tq = self.get_query_tfs(query)
        for doc in docs:
            sim = 0
            sum_wtd = 0
            sum_wtq = 0
            for term in tf_tq.keys():
                if term in self.index and doc in self.index[term]:
                    w_td = self.index[term][doc] # if term is in the index!(gotta check this!)
                else:
                    w_td = 0
                w_tq = tf_tq[term] # 
                if method_list[0] == 'l' and w_td != 0:
                    w_td = 1 + math.log10(self.index[term][doc])
                if method_list[1] == 't' and w_td != 0:
                    w_td *= math.log10(self.N / (self.get_idf(term) + 1))
                if method_list[2] == 'c':
                    sum_wtd += w_td * w_td
                if method_list[3] == 'l':
                    w_td = 1 + math.log10(tf_tq[term])
                if method_list[4] == 't':
                    w_td *= math.log10(self.N / (self.get_idf(term) + 1))
                if method_list[5] == 'c':
                    sum_wtq += w_tq * w_tq
                sim += w_td * w_tq
            if method_list[2] == 'c' and sum_wtd != 0:
                sim /= math.sqrt(sum_wtd)
            if method_list[5] == 'c' and sum_wtq != 0:
                sim /= math.sqrt(sum_wtq)
            docs_to_scores[doc] = sim
        return docs_to_scores

    def compute_socres_with_okapi_bm25(self, query, average_document_field_length, document_lengths):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        
        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        scores = {}

        k1 = 1.2  # Tunable parameter
        b = 0.75  # Tunable parameter

        query_idfs = {}
        for term in query:
            query_idfs[term] = self.get_idf(term)

        for doc in self.get_list_of_documents(query):
            score = 0
            # RSV^BM25 = \sum log(N/idf * [(k+1)tf / k1(1-b + b * dl / avdl) + tf])
            for term in query_idfs.keys():
                idf = math.log10(self.N / (query_idfs[term] + 1))
                tf = 0
                if term in self.index and doc in self.index[term]:
                    tf = self.index[term][doc]
                dl = document_lengths[doc]
                temp = ((k1 + 1)*tf) / (k1 * (1 - b + b * dl / average_document_field_length) + tf)
                score += idf * temp
            scores[doc] = score
        return scores



json_file_path = "/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/index/summaries_index.json"
with open(json_file_path, "r") as file:
    data = json.load(file)
scorer = Scorer(data, 25)
scorer.get_idf("cover")
query = ["redemption", "dark", "knight", "mafia", "father", "god"]
#print(scorer.get_query_tfs("redemption the dark knight redemption"))
#print(scorer.get_list_of_documents(["redemption", "dark", "knight"]))
dic = scorer.compute_scores_with_vector_space_model(query, "lnc.ltc")
for key in dic.keys():
    print(key, dic[key])
