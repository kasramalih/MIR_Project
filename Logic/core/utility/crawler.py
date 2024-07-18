from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json
from preprocess import Preprocessor

class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': 'KasraMalihIMDbCrawler/1.0'
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        self.crawling_threshold = crawling_threshold
        self.not_crawled = list()
        self.crawled = list()
        self.added_ids = list()
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        return URL.split('/')[4]

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        with open('IMDB_crawled.json', 'w') as f:
            f.write(json.dumps(self.crawled, indent=1))
            f.close()

        with open('IMDB_not_crawled.json', 'w') as f:
            f.write(json.dumps(self.not_crawled))
            f.close()

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        """# TODO
        with open('IMDB_crawled.json', 'r') as f:
            self.crawled = None

        with open('IMDB_not_crawled.json', 'w') as f:
            self.not_crawled = None"""
        pass

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        response = get(URL, headers=self.headers)
        return response

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        response = self.crawl(self.top_250_URL)
        soup = BeautifulSoup(response.content, 'html.parser')

        movie_links = soup.find_all('a', {'class': 'ipc-title-link-wrapper'})
        for movie_link in movie_links:
            # since every movie link within site is in form of /title/tt0111161/?ref_=chttp_t_1
            if movie_link['href'].startswith('/title'):
                suffix_url = movie_link['href'].split('/')
                complete_url = 'https://www.imdb.com/title/' + suffix_url[2] + '/'
                self.not_crawled.append(complete_url)
                self.added_ids.append(suffix_url[2])


    def get_imdb_instance(self):
        # python makes a dict instance so no need for this function!
        return {
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
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
            done- replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """

        # help variables

        self.extract_top_250()
        futures = []
        crawled_counter = 0
        """
        self.crawl_page_info('https://www.imdb.com/title/tt0133093')

        return"""

        # WHILE_LOOP_CONSTRAINTS = crawled_counter < self.crawling_threshold
        # NEW_URL = self.not_crawled.pop(0)
        # THERE_IS_NOTHING_TO_CRAWL = check if self.not_crawled is empty or not

        with ThreadPoolExecutor(max_workers=20) as executor:
            while crawled_counter < self.crawling_threshold:
                URL = self.not_crawled.pop(0)
                futures.append(executor.submit(self.crawl_page_info, URL))
                crawled_counter += 1
                print(crawled_counter)
                if not self.not_crawled:
                    wait(futures)
                    futures = []

    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        URL: str
            The URL of the site
        """
        print("new iteration")
        response = self.crawl(URL)
        movie_dict = self.get_imdb_instance()
        self.extract_movie_info(response, movie_dict, URL)
        self.crawled.append(movie_dict)
        related_links = movie_dict['related_links']
        for link in related_links: # type: ignore
            id = self.get_id_from_URL(link)
            if id not in self.added_ids:
                self.not_crawled.append(link)


    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        soup = BeautifulSoup(res.text, 'html.parser')
        movie['id'] = self.get_id_from_URL(URL)
        movie['title'] = self.get_title(soup)
        movie['first_page_summary'] = self.get_first_page_summary(soup)
        movie['release_year'] = self.get_release_year(soup)
        movie['mpaa'] = self.get_mpaa(soup)
        movie['budget'] = self.get_budget(soup)
        movie['gross_worldwide'] = self.get_gross_worldwide(soup)
        movie['directors'] = self.get_director(soup)
        movie['writers'] = self.get_writers(soup)
        movie['stars'] = self.get_stars(soup)
        movie['related_links'] = self.get_related_links(soup)
        movie['genres'] = self.get_genres(soup)
        movie['languages'] = self.get_languages(soup)
        movie['countries_of_origin'] = self.get_countries_of_origin(soup)
        movie['rating'] = self.get_rating(soup)
        summary_link = self.get_summary_link(URL)
        summary_response = get(summary_link, headers=self.headers) # type: ignore
        summary_soup = BeautifulSoup(summary_response.text, 'html.parser')
        movie['summaries'] = self.get_summary(summary_soup)
        movie['synopsis'] = self.get_synopsis(summary_soup)
        review_link = self.get_review_link(URL)
        review_response = get(review_link, headers=self.headers) # type: ignore
        review_soup = BeautifulSoup(review_response.text, 'html.parser')
        movie['reviews'] = self.get_reviews_with_scores(review_soup)

    def get_summary_link(self, url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            summary_url = url.rstrip('/') + '/plotsummary'
            return summary_url
        except:
            print("failed to get summary link")

    def get_review_link(self, url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            review_url = url.rstrip('/') + '/reviews'
            return review_url
        except:
            print("failed to get review link")

    def get_title(self, soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            title = soup.find('span', {'class': 'hero__primary-text'}).text
            return title
        except:
            print("failed to get title")

    def get_first_page_summary(self, soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            first_page_summary = soup.find('span', {'role': 'presentation', 'data-testid': 'plot-xs_to_m'}).text
            return first_page_summary
        except:
            print("failed to get first page summary")

    def get_director(self, soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            # gotta search for Director and Directors!!
            director = soup.find('section', {'data-testid': 'title-cast'}).find(string='Director')
            if director is None:
                directors = soup.find('section', {'data-testid': 'title-cast'}).find(string='Directors')
                directors_cum = directors.findNext().get_text('$', strip=True)
                #split gives a list anyway
                return directors_cum.split("$")
            else:
                return [director.findNext().text]
        except:
            print("failed to get director")

    def get_stars(self, soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            stars = soup.find_all('a', {'data-testid': 'title-cast-item__actor'})
            stars_list = []
            for star in stars:
                stars_list.append(star.text)
            return stars_list
        except:
            print("failed to get stars")

    def get_writers(self, soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            # exactly like directors
            director = soup.find('section', {'data-testid': 'title-cast'}).find(string='Writer')
            if director is None:
                directors = soup.find('section', {'data-testid': 'title-cast'}).find(string='Writers')
                directors_cum = directors.findNext().get_text('$', strip=True)
                # split gives a list anyway
                return directors_cum.split("$")
            else:
                return [director.findNext().text]
        except:
            print("failed to get writers")

    def get_related_links(self, soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            all_div = soup.find_all('a', {'class': "ipc-poster-card__title ipc-poster-card__title--clamp-2 ipc-poster-card__title--clickable"})
            links = []
            for div in all_div:
                if div['href'].startswith('/title'):
                    suffix_url = div['href'].split('/')
                    complete_url = 'https://www.imdb.com/title/' + suffix_url[2] + '/'
                    links.append(complete_url)
            return links
        except:
            print("failed to get related links")

    def get_summary(self, soup):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            sum_div = soup.find('div', {'data-testid': "sub-section-summaries"})
            all_sums = sum_div.find_all('li', {'data-testid': "list-item"})
            summary_list = []
            for summary in all_sums:
                summary_list.append(summary.text)
            return summary_list
        except:
            print("failed to get summary")

    def get_synopsis(self, soup):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            synopsis = soup.find('div', {'data-testid': "sub-section-synopsis"}).text
            return [synopsis]
        except:
            print("failed to get synopsis")

    def get_reviews_with_scores(self, soup):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            counter = 0
            all_reviews = soup.find_all('div', {'class': 'lister-item-content'})
            review_list = [['just in case', '10']]
            for rev in all_reviews:
                if counter > 19:
                    break
                score = rev.find('span', {'class': "point-scale"})
                if score is None:
                    score = 'no score!'
                else:
                    score = score.previousSibling.text
                review = rev.find('div', {'class': "text show-more__control"})
                if review is None:
                    review_prime = rev.find('div', {'class': "text show-more__control clickable"})
                    if review_prime is not None:
                        rs = [review_prime.text, score]
                        counter += 1
                        review_list.append(rs)
                else:
                    rs = [review.text, score]
                    review_list.append(rs)
                    counter += 1
            if review_list:
                return review_list
        except:
            print("failed to get reviews")

    def get_genres(self, soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            # capped to 3 genres for more we gon need selenium
            genres = soup.select_one('div.ipc-chip-list__scroller')
            genre_list = []
            for genre in genres.contents:
                genre_list.append(genre.text)
            return genre_list
        except:
            print("Failed to get generes")

    def get_rating(self, soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            rating = soup.find('span', {'class': 'sc-bde20123-1 cMEQkK'}).text
            return rating
        except:
            print("failed to get rating")

    def get_mpaa(self, soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            temp = soup.find_all('a', {'class': 'ipc-link ipc-link--baseAlt ipc-link--inherit-color'})
            """
            0- Cast & crew
            1- User reviews
            2- Trivia
            3- FAQ
            4- IMDbPro
            5- 2008
            6- PG-13
            """
            return temp[6].text
        except:
            print("failed to get mpaa")

    def get_release_year(self, soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            temp = soup.find_all('a', {'class': 'ipc-link ipc-link--baseAlt ipc-link--inherit-color'})
            return temp[5].text
        except:
            print("failed to get release year")

    def get_languages(self, soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        # like directors again!
        try:
            language = soup.find('section', {'data-testid': 'Details'}).find(string='Language')
            if language is None:
                languages = soup.find('section', {'data-testid': 'Details'}).find(string='Languages')
                all_languages = languages.findNext().get_text('$', strip=True)
                return all_languages.split("$")
            else:
                return [language.findNext().text]
        except:
            print("failed to get languages")

    def get_countries_of_origin(self, soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            country = soup.find('section', {'data-testid': 'Details'}).find(string='Country of origin')
            if country is None:
                countries = soup.find('section', {'data-testid': 'Details'}).find(string='Countries of origin')
                all_countries = countries.findNext().get_text('$', strip=True)
                return all_countries.split("$")
            else:
                return [country.findNext().text]
        except:
            print("failed to get countries of origin")

    def get_budget(self, soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            budget = soup.find('li', {'data-testid': 'title-boxoffice-budget'})
            # budget.text                    | budget.text[6:]
            # Budget$185,000,000 (estimated) | $185,000,000 (estimated)
            return budget.text[6:]
        except:
            print("failed to get budget")

    def get_gross_worldwide(self, soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            gross_worldwide = soup.find('li', {'data-testid': 'title-boxoffice-cumulativeworldwidegross'}).text[15:]
            return gross_worldwide
        except:
            print("failed to get gross worldwide")


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=1000)
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()
    """    
    json_file_path = "/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/IMDB_crawled.json"
    with open(json_file_path, "r") as file:
        data = json.load(file)
    preprocessor = Preprocessor(data)
    preprocessed_data = preprocessor.preprocess()
    with open('preprocessed_data.json', 'w') as f:
        f.write(json.dumps(preprocessed_data, indent=1))
        f.close()
    """

if __name__ == '__main__':
    main()
