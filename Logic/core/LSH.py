import numpy as np
import itertools
import random


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = set()
        words = document.split()
        num_words = len(words)
        for i in range(num_words - k + 1):
            shingle = ' '.join(words[i:i+k])
            shingles.add(shingle)
        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        shingled_documents = [self.shingle_document(doc) for doc in self.documents]
        shingles = sorted(set().union(*shingled_documents))
        print(shingles)
        num_docs = len(self.documents)
        num_shingles = len(shingles)
        characteristic_matrix = np.zeros((num_shingles, num_docs), dtype=int)
        for i, shingle in enumerate(shingles):
            for j, doc in enumerate(shingled_documents):
                if shingle in doc:
                    characteristic_matrix[i, j] = 1
        return characteristic_matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        characteristic_matrix = self.build_characteristic_matrix()
        num_shingles, num_docs= characteristic_matrix.shape
        signatures = np.full((self.num_hashes, num_docs), np.inf)
        # maybe I need to change the permutation to a hash function!
        hashes = [np.random.permutation(num_shingles) for _ in range(self.num_hashes)]
        print(hashes)
        for i in range(self.num_hashes):
            for j in range(num_docs):
                for k in range(num_shingles):
                    if characteristic_matrix[k, j] == 1:
                        signatures[i,j] = min(signatures[i,j], hashes[i][k])
        return signatures

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO band hashes being overwritten, fix it!
        buckets = {}
        num_hashes, num_docs = signature.shape
        # split each doc(cols in signature matrix) to bands
        # rows per band shall be calculated not given!
        rpb = int(num_hashes / bands)
        for b in range(bands):
            # Hash each band
            band_start = b * rpb
            band_end = (b + 1) * rpb
            band_hashes = [hash(tuple(signature[band_start:band_end, d])) for d in range(num_docs)]
        for i, doc_hash in enumerate(band_hashes):
            if doc_hash not in buckets:
                buckets[doc_hash] = []
            buckets[doc_hash].append(i)
        
        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO
        return self.lsh_buckets(self.min_hash_signature(), bands=10, rows_per_band=10)

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        jaccard_score = intersection / union if union != 0 else 0
        return jaccard_score

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)


docs = [
    'kasra khare',
    'kir to zendegi',
    'kir e khar',
    'khar to zendegi'
    ]
minHashLSH = MinHashLSH(docs,3)
print(minHashLSH.build_characteristic_matrix()) # tested and ok
print(minHashLSH.min_hash_signature()) # tested and ok