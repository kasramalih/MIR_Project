import json


class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()
        
        for i in range(len(word) - k + 1):
            shingle = word[i:i + k]
            shingles.add(shingle)

        return shingles
    
    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        jaccard_score = intersection / union if union != 0 else 0
        return jaccard_score

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()

        for doc in all_documents:
            # change this so you read from indexed summaries, much faster, but words are stemmed!!
            text = doc['summaries'] # just searching for words in summaries ... maybe added next fields
            if text is not None:
                for element in text:
                    for word in element.split():
                        if word not in word_counter.keys():
                            word_counter[word] = 0
                            all_shingled_words[word] = self.shingle_word(word)
                        word_counter[word] += 1
                
        return all_shingled_words, word_counter
    
    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = list()

        misspelled_shingles = self.shingle_word(word)
        temp_list = list()
        for key in self.all_shingled_words.keys():
            candidate_shingles = self.all_shingled_words[key]
            score = self.jaccard_score(misspelled_shingles, candidate_shingles)
            if len(temp_list) < 5:
                temp_list.append([score, key])
                temp_list = sorted(temp_list, reverse = True)
            # elif score > temp_list[4][0]:
            else:
                temp_list.append([score, key])
                temp_list = sorted(temp_list, reverse = True)
                temp_list.pop()
        print(temp_list)
        for sk in temp_list:
            top5_candidates.append(sk[1])
        return top5_candidates
    
    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : str
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = ""

        for word in query.split():
            top_5_candids = self.find_nearest_words(word)
            print(word, '\n',top_5_candids)
            final_result += top_5_candids[0] + ''

        return final_result


json_file_path = "/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/IMDB_crawled.json"
with open(json_file_path, "r") as file:
    data = json.load(file)
spell = SpellCorrection(data)
query = 'the darkh knjght ank joher'
res = spell.spell_check(query)