import streamlit as st
import sys

sys.path.append('/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/Logic')
import utils
import time
from enum import Enum
import random
import streamlit_authenticator as stauth
from streamlit_star_rating import st_star_rating
import yaml
from core.utility.snippet import Snippet
from core.link_analysis.analyzer import LinkAnalyzer
from core.indexer.index_reader import Index_reader, Indexes

snippet_obj = Snippet()


class color(Enum):
    RED = "#FF0000"
    GREEN = "#00FF00"
    BLUE = "#0000FF"
    YELLOW = "#FFFF00"
    # WHITE = "#FFFFFF"
    CYAN = "#00FFFF"
    MAGENTA = "#FF00FF"

##################################################################################################
#movie recommendation: SINCE MOVIE IDS ARE NOT THE SAME I CAN NOTTTTT FUCK!
##################################################################################################

# Load config for authentication
with open('/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/UI/config.yaml') as file:
    config = yaml.safe_load(file)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

def get_top_x_movies_by_rank(x: int, results: list):
    path = "/Users/kianamalihi/Desktop/MIR_PROJECT/MIR_Project/index"  # Link to the index folder
    document_index = Index_reader(path, Indexes.DOCUMENTS)
    corpus = []
    root_set = []
    for movie_id, movie_detail in document_index.index.items():
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        corpus.append({"id": movie_id, "title": movie_title, "stars": stars})

    for element in results:
        movie_id = element[0]
        movie_detail = document_index.index[movie_id]
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        root_set.append({"id": movie_id, "title": movie_title, "stars": stars})
    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=x)
    return actors, movies


def get_summary_with_snippet(movie_info, query):
    summary = movie_info["first_page_summary"]
    snippet, not_exist_words = snippet_obj.find_snippet(movie_info, query)
    if "***" in snippet:
        summary = snippet
        snippet = snippet.split()
        for i in range(len(snippet)):
            current_word = snippet[i]
            if current_word.startswith("***") and current_word.endswith("***"):
                current_word_without_star = current_word[3:-3]
                summary = summary.lower().replace(
                    current_word_without_star,
                    f"<b><font size='4' color={random.choice(list(color)).value}>{current_word_without_star}</font></b>",
                )
    return summary


def search_time(start, end):
    st.success("Search took: {:.6f} milli-seconds".format((end - start) * 1e3))


def function_to_run_on_click(value, title):
    # for rate history:
    if value != 0:
        if "user" in st.session_state:
            if "rate_history" not in st.session_state:
                st.session_state["rate_history"] = {}
            if st.session_state["user"] not in st.session_state["rate_history"]:
                st.session_state["rate_history"][st.session_state["user"]] = {}
            print(title, value)
            st.session_state["rate_history"][st.session_state["user"]][title] = value

def search_handling(
    search_button,
    search_term,
    search_max_num,
    search_weights,
    search_method,
    unigram_smoothing,
    alpha,
    lamda,
    filter_button,
    num_filter_results,
    username
):
    if filter_button:
        if "search_results" in st.session_state:
            top_actors, top_movies = get_top_x_movies_by_rank(
                num_filter_results, st.session_state["search_results"]
            )
            st.markdown(f"**Top {num_filter_results} Actors:**")
            actors_ = ", ".join(top_actors)
            st.markdown(
                f"<span style='color:{random.choice(list(color)).value}'>{actors_}</span>",
                unsafe_allow_html=True,
            )
            st.divider()

        st.markdown(f"**Top {num_filter_results} Movies:**")
        for i in range(len(top_movies)):
            card = st.columns([3, 1])
            info = utils.get_movie_by_id(top_movies[i], utils.movies_dataset)
            with card[0].container():
                # print(info)
                st.title(info[0]["title"])
                st.markdown(f"[Link to movie]({info[0]['URL']})")
                st.markdown(
                    f"<b><font size = '4'>Summary:</font></b> {get_summary_with_snippet(info[1], search_term)}",
                    unsafe_allow_html=True,
                )

            with st.container():
                st.markdown("**Directors:**")
                num_authors = len(info[0]["directors"])
                for j in range(num_authors):
                    st.text(info[0]["directors"][j])

            with st.container():
                st.markdown("**Stars:**")
                num_authors = len(info[0]["stars"])
                stars = "".join(star + ", " for star in info[0]["stars"])
                st.text(stars[:-2])

                topic_card = st.columns(1)
                with topic_card[0].container():
                    st.write("Genres:")
                    num_topics = len(info[0]["genres"])
                    for j in range(num_topics):
                        st.markdown(
                            f"<span style='color:{random.choice(list(color)).value}'>{info[0]['genres'][j]}</span>",
                            unsafe_allow_html=True,
                        )
            with card[1].container():
                st.image(info[0]["Image_URL"], use_column_width=True)

            st.divider()
        return
    
    #if search_button:
    if search_button == 'search':
        corrected_query = utils.correct_text(search_term)
        # corrected_query = utils.correct_text(search_term, utils.all_documents)
        if corrected_query != search_term:
            st.warning(f"Your search terms were corrected to: {corrected_query}")
            search_term = corrected_query

        with st.spinner("Searching..."):
            time.sleep(0.5)  # for showing the spinner! (can be removed)
            start_time = time.time()
            result = utils.search(
                search_term,
                search_max_num,
                search_method,
                search_weights,
                smoothing_method = unigram_smoothing,
                alpha=alpha,
                lamda=lamda,
            )
            if "search_results" in st.session_state:
                st.session_state["search_results"] = result
            print(f"Result: {result}")
            # for search history:
            if "user" in st.session_state:
                if "search_history" not in st.session_state:
                    st.session_state["search_history"] = {}
                if st.session_state["user"] not in st.session_state["search_history"]:
                    st.session_state["search_history"][st.session_state["user"]] = []
                st.session_state["search_history"][st.session_state["user"]].append(search_term)

            end_time = time.time()
            if len(result) == 0:
                st.warning("No results found!")
                return

            search_time(start_time, end_time)

            for i in range(len(result)):
                card = st.columns([3, 1])
                info = utils.get_movie_by_id(result[i][0], utils.movies_dataset)
                with card[0].container():
                    st.title(info[0]["title"])
                    st.markdown(f"[Link to movie]({info[0]['URL']})")
                    st.write(f"Relevance Score: {result[i][1]}")
                    st.markdown(
                        f"<b><font size = '4'>Summary:</font></b> {get_summary_with_snippet(info[1], search_term)}",
                        unsafe_allow_html=True,
                    )

                with st.container():
                    st.markdown("**Directors:**")
                    num_authors = len(info[0]["directors"])
                    for j in range(num_authors):
                        st.text(info[0]["directors"][j])

                with st.container():
                    st.markdown("**Stars:**")
                    num_authors = len(info[0]["stars"])
                    stars = "".join(star + ", " for star in info[0]["stars"])
                    st.text(stars[:-2])

                    topic_card = st.columns(1)
                    with topic_card[0].container():
                        st.write("Genres:")
                        num_topics = len(info[0]["genres"])
                        for j in range(num_topics):
                            st.markdown(
                                f"<span style='color:{random.choice(list(color)).value}'>{info[0]['genres'][j]}</span>",
                                unsafe_allow_html=True,
                            )
                with card[1].container():
                    st.image(info[0]["Image_URL"], use_column_width=True)
                    default_val = 0
                    
                    if "rate_history" in st.session_state and username in st.session_state["rate_history"] and info[0]["title"] in st.session_state["rate_history"][username]:                    
                        default_val = st.session_state["rate_history"][username][info[0]["title"]]
                    
                    stars = st_star_rating(
                        "your rating", maxValue=5, defaultValue=default_val, size=20,
                        key=info[0]["Image_URL"],
                        on_click=function_to_run_on_click,
                        on_click_kwargs={'title': info[0]["title"]})
            st.divider()

        st.session_state["search_results"] = result
        if "filter_state" in st.session_state:
            st.session_state["filter_state"] = (
                "search_results" in st.session_state
                and len(st.session_state["search_results"]) > 0
            )


def main():
    st.title("Search Engine")
    st.write(
        "This is a simple search engine for IMDB movies. You can search through IMDB dataset and find the most relevant movie to your search terms."
    )
    st.markdown(
        '<span style="color:yellow">Developed By: MIR Team at Sharif University</span>',
        unsafe_allow_html=True,
    )

    authenticator_name = st.sidebar.radio(" ", ["Login", "Sign Up"])

    if authenticator_name == "Login":
        name, authentication_status, username = authenticator.login('main', fields = {'Form name': 'Login'})

        if authentication_status:
            st.session_state["user"] = username
            authenticator.logout("Logout", "sidebar")
            st.sidebar.title(f"Welcome {username}")

            # Display user search history
            with st.sidebar.expander("**Search History**"):
                if "search_history" in st.session_state and username in st.session_state["search_history"]:
                    # st.sidebar.markdown("**Search History:**")
                    shown_in_search = []
                    for search in st.session_state["search_history"][username]:
                        if search != '':
                            if search not in shown_in_search:
                                st.markdown(f"{search}")
                                shown_in_search.append(search)
            
            with st.sidebar.expander("**Your Ratings**"):
            # Display user rate history
                if "rate_history" in st.session_state and username in st.session_state["rate_history"]:                    
                    for key,value in st.session_state["rate_history"][username].items():
                        st.markdown(f"{key}: {value}")


        elif authentication_status is False:
            st.error('Username/password is incorrect')
        elif authentication_status is None:
            st.warning('Please enter your username and password')


    search_term = st.text_input("Seacrh Term")
    with st.expander("Advanced Search"):
        search_max_num = st.number_input(
            "Maximum number of results", min_value=5, max_value=100, value=10, step=5
        )
        weight_stars = st.slider(
            "Weight of stars in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )

        weight_genres = st.slider(
            "Weight of genres in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )

        weight_summary = st.slider(
            "Weight of summary in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )
        slider_ = st.slider("Select the number of top movies to show", 1, 10, 5)

        search_weights = [weight_stars, weight_genres, weight_summary]
        search_method = st.selectbox(
            "Search method", ("ltn.lnn", "ltc.lnc", "OkapiBM25", "unigram")
        )

        unigram_smoothing = None
        alpha, lamda = None, None
        if search_method == "unigram":
            unigram_smoothing = st.selectbox(
                "Smoothing method",
                ("naive", "bayes", "mixture"),
            )
            if unigram_smoothing == "bayes":
                alpha = st.slider(
                    "Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )
            if unigram_smoothing == "mixture":
                alpha = st.slider(
                    "Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )
                lamda = st.slider(
                    "Lambda",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )

    if "search_results" not in st.session_state:
        st.session_state["search_results"] = []

    # search_button = st.button("Search!")
    search_button = st.radio("search", ['search'])
    filter_button = st.button("Filter movies by ranking")

    search_handling(
        search_button,
        search_term,
        search_max_num,
        search_weights,
        search_method,
        unigram_smoothing,
        alpha,
        lamda,
        filter_button,
        slider_,
        username,
    )


if __name__ == "__main__":
    main()
