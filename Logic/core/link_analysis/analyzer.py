from .graph import LinkGraph
# from ..indexer.indexes_enum import Indexes
# from ..indexer.index_reader import Index_reader
import json
class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = []
        self.authorities = []
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        # creating graph of root base! all movies are hubs and all stars in them are authority at first!
        for movie in self.root_set:
            movie_id = movie['id']
            self.graph.add_node(movie_id)
            self.hubs.append(movie_id)
            for star in movie['stars']:
                self.graph.add_node(star)
                self.authorities.append(star)
                self.graph.add_edge(star, movie_id)  # Actors point to movies
                self.graph.add_edge(movie_id, star)  # Movies point to actors

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in corpus:
            movie_id = movie['id']
            if movie_id not in self.graph.graph:
                self.graph.add_node(movie_id)
            for star in movie['stars']:
                if star not in self.graph.graph:
                    self.graph.add_node(star)
                self.graph.add_edge(star, movie_id)  # Actors point to movies
                self.graph.add_edge(movie_id, star)  # Movies point to actors
                if movie_id in self.hubs or star in self.authorities:
                    if movie_id not in self.hubs:
                        self.hubs.append(movie_id)
                    if star not in self.authorities:
                        self.authorities.append(star)

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        a_s = {node: 1.0 for node in self.authorities}
        h_s = {node: 1.0 for node in self.hubs}

        for iter in range(num_iteration):
            new_a_s = {node: 0.0 for node in a_s}
            new_h_s = {node: 0.0 for node in h_s}

            for node in self.hubs:
                for successor in self.graph.get_successors(node):
                    if successor in new_a_s.keys():
                        new_a_s[successor] += h_s[node]

            for node in self.authorities:
                for predecessor in self.graph.get_predecessors(node):
                    if predecessor in new_h_s.keys():
                        new_h_s[predecessor] += a_s[node]

            norm = max(new_a_s.values())
            if norm > 0:
                for node in new_a_s:
                    new_a_s[node] /= norm

            norm = max(new_h_s.values())
            if norm > 0:
                for node in new_h_s:
                    new_h_s[node] /= norm

            a_s, new_a_s = new_a_s, a_s
            h_s, new_h_s = new_h_s, h_s

        top_authorities = sorted(a_s.items(), key=lambda item: item[1], reverse=True)[:max_result]
        top_hubs = sorted(h_s.items(), key=lambda item: item[1], reverse=True)[:max_result]

        top_actors = [node for node, score in top_authorities if node in self.authorities]
        top_movies = [node for node, score in top_hubs if node in self.hubs]

        return top_actors, top_movies


if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    json_file_path = "/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/IMDB_crawled.json"
    with open(json_file_path, "r") as file:
        data = json.load(file)
    corpus = data    # TODO: it shoud be your crawled data
    root_set = [data[0], data[1], data[2], data[3]]   # TODO: it shoud be a subset of your corpus

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=10)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
