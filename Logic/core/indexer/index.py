from operator import index
import time
import os
import json
import copy
from indexes_enum import Indexes


class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
        }

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {}
        for doc in self.preprocessed_documents:
            id = doc['id']
            current_index[id] = doc
        return current_index

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        star_based_index = {}
        for doc in self.preprocessed_documents:
            stars = doc['stars']
            doc_id = doc['id']

            for star in stars:
                if star not in star_based_index:
                    star_based_index[star] = {}
                if doc_id not in star_based_index[star]:
                    star_based_index[star][doc_id] = 0
                star_based_index[star][doc_id] += 1
        
        return star_based_index

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        genre_based_index = {}
        for doc in self.preprocessed_documents:
            genres = doc['genres']
            doc_id = doc['id']
            if genres is None:
                continue
            for genre in genres:
                if genre not in genre_based_index:
                    genre_based_index[genre] = {}
                if doc_id not in genre_based_index[genre]:
                    genre_based_index[genre][doc_id] = 0
                genre_based_index[genre][doc_id] += 1
        
        return genre_based_index

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        summary_based_index = {}
        for doc in self.preprocessed_documents:
            summaries = doc['summaries']
            doc_id = doc['id']
            if summaries is not None:
                for summary in summaries:
                    if summary not in summary_based_index:
                        summary_based_index[summary] = {}
                    if doc_id not in summary_based_index[summary]:
                        summary_based_index[summary][doc_id] = 0
                    summary_based_index[summary][doc_id] += 1
        
        return summary_based_index

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """
        posting_list = []
        try:
            if index_type == 'documents':
                doc_dict = self.index['documents']
                for key in doc_dict.keys():
                    if word in doc_dict[key]:
                        posting_list.append(key)
            elif index_type == 'stars':
                star_to_doc_dict = self.index['stars'] #  {term: {document_id: tf}}
                for key in star_to_doc_dict.keys():
                    if key == word:
                        for doc_id in star_to_doc_dict[key]:
                            if doc_id not in posting_list:
                                posting_list.append(doc_id)
            elif index_type == 'genres':
                genre_to_doc_dict = self.index['genres'] #  {term: {document_id: tf}}
                for key in genre_to_doc_dict.keys():
                    if key == word:
                        for doc_id in genre_to_doc_dict[key]:
                            if doc_id not in posting_list:
                                posting_list.append(doc_id)
            elif index_type == 'summaries':
                summary_to_doc_dict = self.index['summaries'] #  {term: {document_id: tf}}
                for key in summary_to_doc_dict.keys():
                    if key == word:
                        for doc_id in summary_to_doc_dict[key]:
                            if doc_id not in posting_list:
                                posting_list.append(doc_id)
            else:
                print('WRONG INDEX TYPE!')
        except:
            return posting_list
        
        return posting_list

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """
        doc_id = document['id']
        stars = document['stars']
        genres = document['genres']
        summaries = document['summaries']

        #doc index
        doc_dict = self.index['documents']
        doc_dict[doc_id] = document
        self.index['documents'] = doc_dict

        #stars index
        star_based_index = self.index['stars']
        for star in stars:
            if star not in star_based_index:
                star_based_index[star] = {}
            if doc_id not in star_based_index[star]:
                star_based_index[star][doc_id] = 0
            star_based_index[star][doc_id] += 1
        self.index['stars'] = star_based_index
        #genres index
        genre_based_index = self.index['genres']
        for genre in genres:
                if genre not in genre_based_index:
                    genre_based_index[genre] = {}
                if doc_id not in genre_based_index[genre]:
                    genre_based_index[genre][doc_id] = 0
                genre_based_index[genre][doc_id] += 1
        self.index['genres'] = genre_based_index
        #summaries index
        summary_based_index = self.index['summaries']
        for summary in summaries:
            if summary not in summary_based_index:
                summary_based_index[summary] = {}
            if doc_id not in summary_based_index[summary]:
                summary_based_index[summary][doc_id] = 0
            summary_based_index[summary][doc_id] += 1
        self.index['summaries'] = summary_based_index

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """

        doc_dict = self.index['documents']
        del doc_dict[document_id]

        star_based_index = self.index['stars']
        for star in star_based_index.keys():
            temp_dict = star_based_index[star]
            if document_id in temp_dict.keys:
                del temp_dict[document_id]

        genre_based_index = self.index['genres']
        for genre in genre_based_index.keys():
            temp_dict = genre_based_index[genre]
            if document_id in temp_dict.keys:
                del temp_dict[document_id]

        summary_based_index = self.index['summaries']
        for summary in summary_based_index.keys():
            temp_dict = summary_based_index[summary]
            if document_id in temp_dict.keys:
                del temp_dict[document_id]

    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['tim', 'henry'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return
        print(index_after_add[Indexes.STARS.value]['tim'])
        # print(index_before_add[Indexes.STARS.value]['tim']) TABIE E KE QABL ADD IN NABASHE DIGE IN CHE TEST MASKHARE I HAST!
        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(set(index_before_add[Indexes.STARS.value]['tim']))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        if (set(index_after_add[Indexes.STARS.value]['henry']).difference(set(index_before_add[Indexes.STARS.value]['henry']))
                != {dummy_document['id']}):
            print('Add is incorrect, henry')
            return
        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(set(index_before_add[Indexes.GENRES.value]['drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(set(index_before_add[Indexes.SUMMARIES.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str, index_name: str):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, stars, genres, summaries)
        """

        if not os.path.exists(path):
            os.makedirs(path)

        if index_name not in self.index:
            raise ValueError('Invalid index name')

        with open(path+'/'+index_name+'_index.json', 'w') as f:
            f.write(json.dumps(self.index[index_name], indent=1))
            f.close()

        return

    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """

        with open(path, "r") as file:
            data = json.load(file)
        return data

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """

        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field:
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        # based on your implementation, you may need to change the following line
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)): # type: ignore
            print('Indexing is correct')

            if implemented_time < brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False

# Run the class with needed parameters, then run check methods and finally report the results of check methods

json_file_path = "/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/preprocessed_data.json"
with open(json_file_path, "r") as file:
    data = json.load(file)
indexer = Index(data)
indexer.store_index('/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/index', 'documents')
indexer.store_index('/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/index', 'stars')
indexer.store_index('/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/index', 'genres')
indexer.store_index('/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/index', 'summaries')

# indexer.check_add_remove_is_correct()
# print(indexer.check_if_index_loaded_correctly('genres', indexer.load_index('/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/index/genres_index.json')))
#cindexer.check_if_indexing_is_good('genres', 'crime')
print('done')