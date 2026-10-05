import streamlit as st
import pandas as pd
import ast
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0b0b0f;
    color: #ffffff;
}

[data-testid="stHeader"] {
    background-color: #0b0b0f;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Header */

.main-title {
    font-size: 52px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
    color: white;
}

.subtitle {
    text-align: center;
    color: #999999;
    font-size: 18px;
    margin-bottom: 35px;
}

/* Section titles */

.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 35px;
    margin-bottom: 20px;
    color: white;
}

/* Cards */

.movie-card {
    background: #15151c;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 20px;
    border: 1px solid #252530;
}

.info-card {
    background: #15151c;
    border-radius: 14px;
    padding: 22px;
    border: 1px solid #252530;
    height: 100%;
}

.feature-card {
    background: #15151c;
    border-radius: 14px;
    padding: 25px;
    border: 1px solid #252530;
    min-height: 180px;
}

/* Movie title */

.movie-title {
    font-size: 22px;
    font-weight: 700;
    color: white;
    margin-bottom: 8px;
}

.movie-meta {
    color: #aaaaaa;
    font-size: 14px;
}

/* Recommendation score */

.score {
    font-size: 18px;
    font-weight: 700;
    color: #4ade80;
}

/* Progress bar */

.progress-container {
    background: #292933;
    border-radius: 10px;
    height: 8px;
    width: 100%;
    margin-top: 8px;
    margin-bottom: 10px;
}

.progress-bar {
    background: #e50914;
    height: 8px;
    border-radius: 10px;
}

/* Tags */

.genre-tag {
    display: inline-block;
    background: #252530;
    color: #dddddd;
    padding: 5px 10px;
    border-radius: 20px;
    margin-right: 5px;
    margin-bottom: 5px;
    font-size: 12px;
}

/* Footer */

.footer {
    text-align: center;
    color: #777777;
    margin-top: 50px;
    padding-top: 25px;
    border-top: 1px solid #252530;
}

/* Select box */

div[data-baseweb="select"] > div {
    background-color: #15151c;
    border-color: #33333d;
}

/* Buttons */

.stButton > button {
    background-color: #e50914;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #b20710;
    color: white;
}

/* Expander */

[data-testid="stExpander"] {
    background-color: #15151c;
    border: 1px solid #252530;
    border-radius: 12px;
}

/* Divider */

hr {
    border-color: #252530;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎬 CineMatch AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Movie Recommendation System</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD MOVIE DATA
# =========================================================

@st.cache_data(show_spinner=False)
def load_movies():

    movies_path = "tmdb_5000_movies.csv"
    credits_path = "tmdb_5000_credits.csv"

    # Check files
    if not os.path.exists(movies_path):
        st.error(f"File not found: {movies_path}")
        st.stop()

    if not os.path.exists(credits_path):
        st.error(f"File not found: {credits_path}")
        st.stop()

    # Read CSV files
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # -----------------------------------------------------
    # Fix title column
    # -----------------------------------------------------

    if "title" not in movies.columns:

        if "original_title" in movies.columns:
            movies["title"] = movies["original_title"]

        elif "movie_title" in movies.columns:
            movies["title"] = movies["movie_title"]

    # -----------------------------------------------------
    # Fix movie ID
    # -----------------------------------------------------

    if "id" not in movies.columns:

        if "movie_id" in movies.columns:
            movies["id"] = movies["movie_id"]

        else:
            st.error("Movie ID column was not found.")

            st.write(
                "Movie CSV columns:",
                list(movies.columns)
            )

            st.stop()

    # -----------------------------------------------------
    # Merge credits
    # -----------------------------------------------------

    if "movie_id" in credits.columns:

        movies = movies.merge(
            credits,
            left_on="id",
            right_on="movie_id",
            how="left",
            suffixes=("", "_credits")
        )

    else:

        st.warning(
            "Credits CSV does not contain movie_id. "
            "Recommendation data may be incomplete."
        )

    # -----------------------------------------------------
    # Final title check
    # -----------------------------------------------------

    if "title" not in movies.columns:

        st.error(
            "The movie dataset does not contain a usable title column."
        )

        st.write(
            "Available movie columns:",
            list(movies.columns)
        )

        st.stop()

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    if "overview" not in movies.columns:
        movies["overview"] = ""

    if "genres" not in movies.columns:
        movies["genres"] = "[]"

    if "keywords" not in movies.columns:
        movies["keywords"] = "[]"

    if "cast" not in movies.columns:
        movies["cast"] = "[]"

    if "crew" not in movies.columns:
        movies["crew"] = "[]"

    if "vote_average" not in movies.columns:
        movies["vote_average"] = 0

    # Clean title
    movies["title"] = movies["title"].fillna("").astype(str)

    # Remove movies without titles
    movies = movies[
        movies["title"].str.strip() != ""
    ].reset_index(drop=True)

    return movies


movies = load_movies()


# =========================================================
# LOAD LOCAL POSTERS
# =========================================================

@st.cache_data(show_spinner=False)
def load_posters():

    poster_folder = "posters"

    poster_lookup = {}

    if not os.path.exists(poster_folder):
        return poster_lookup

    for filename in os.listdir(poster_folder):

        # Accept JPG, JPEG and PNG
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            movie_id = os.path.splitext(
                filename
            )[0]

            try:

                movie_id = int(movie_id)

                poster_lookup[movie_id] = os.path.join(
                    poster_folder,
                    filename
                )

            except:

                continue

    return poster_lookup


poster_lookup = load_posters()


# =========================================================
# GET LOCAL POSTER
# =========================================================

def get_poster(movie_id):

    try:
        movie_id = int(movie_id)
    except:
        return None

    poster_path = poster_lookup.get(movie_id)

    if poster_path and os.path.exists(poster_path):
        return poster_path

    return None


# =========================================================
# PARSE JSON-LIKE DATA
# =========================================================

def parse_names(value):

    if pd.isna(value):
        return []

    try:

        data = ast.literal_eval(str(value))

        if isinstance(data, list):

            names = []

            for item in data:

                if isinstance(item, dict):

                    name = item.get("name", "")

                    if name:
                        names.append(str(name))

            return names

        return []

    except:

        return []


# =========================================================
# GET TOP 3 CAST
# =========================================================

def get_top_cast(value):

    if pd.isna(value):
        return []

    try:

        data = ast.literal_eval(str(value))

        if isinstance(data, list):

            cast_names = []

            for item in data[:3]:

                if isinstance(item, dict):

                    name = item.get("name", "")

                    if name:
                        cast_names.append(str(name))

            return cast_names

        return []

    except:

        return []


# =========================================================
# GET DIRECTOR
# =========================================================

def get_director(value):

    if pd.isna(value):
        return ""

    try:

        data = ast.literal_eval(str(value))

        if isinstance(data, list):

            for item in data:

                if isinstance(item, dict):

                    if item.get("job") == "Director":

                        return str(
                            item.get("name", "")
                        )

        return ""

    except:

        return ""


# =========================================================
# PREPARE DATA
# =========================================================

@st.cache_data(show_spinner=False)
def prepare_data(df):

    data = df.copy()

    # Safety checks
    if "title" not in data.columns:
        st.error("Title column is missing.")
        st.write("Available columns:", list(data.columns))
        st.stop()

    # Make sure required columns exist
    if "overview" not in data.columns:
        data["overview"] = ""

    if "genres" not in data.columns:
        data["genres"] = "[]"

    if "keywords" not in data.columns:
        data["keywords"] = "[]"

    if "cast" not in data.columns:
        data["cast"] = "[]"

    if "crew" not in data.columns:
        data["crew"] = "[]"

    # Parse movie information
    data["genres"] = data["genres"].apply(
        parse_names
    )

    data["keywords"] = data["keywords"].apply(
        parse_names
    )

    data["cast"] = data["cast"].apply(
        get_top_cast
    )

    data["director"] = data["crew"].apply(
        get_director
    )

    # Convert lists into text
    data["genres_text"] = data["genres"].apply(
        lambda x: " ".join(x)
    )

    data["keywords_text"] = data["keywords"].apply(
        lambda x: " ".join(x)
    )

    data["cast_text"] = data["cast"].apply(
        lambda x: " ".join(x)
    )

    # Clean overview
    data["overview"] = (
        data["overview"]
        .fillna("")
        .astype(str)
    )

    data["director"] = (
        data["director"]
        .fillna("")
        .astype(str)
    )

    # -----------------------------------------------------
    # Create combined AI tags
    # -----------------------------------------------------

    data["tags"] = (
        data["overview"]
        + " "
        + data["genres_text"]
        + " "
        + data["keywords_text"]
        + " "
        + data["cast_text"]
        + " "
        + data["director"]
    )

    data["tags"] = (
        data["tags"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    return data.reset_index(drop=True)


# =========================================================
# CREATE TF-IDF MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def create_tfidf(data):

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    vectors = vectorizer.fit_transform(
        data["tags"]
    )

    return vectors


# =========================================================
# CREATE COSINE SIMILARITY
# =========================================================

@st.cache_data(show_spinner=False)
def create_similarity_matrix(_vectors):

    return cosine_similarity(
        _vectors
    )


# =========================================================
# PREPARE AI MODEL
# =========================================================

with st.spinner("🎬 Preparing CineMatch AI..."):

    movies = prepare_data(movies)

    vectors = create_tfidf(movies)

    similarity = create_similarity_matrix(
        vectors
    )


# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def recommend_movies(
    movie_title,
    number_of_movies=5
):

    # Create title → index mapping
    movie_indices = pd.Series(
        movies.index,
        index=movies["title"]
    ).drop_duplicates()

    # Movie doesn't exist
    if movie_title not in movie_indices:
        return pd.DataFrame()

    index = movie_indices[movie_title]

    # Get similarity scores
    similarity_scores = list(
        enumerate(
            similarity[index]
        )
    )

    # Sort highest similarity first
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove selected movie
    similarity_scores = similarity_scores[
        1:number_of_movies + 1
    ]

    # Movie indexes
    movie_indices_result = [
        item[0]
        for item in similarity_scores
    ]

    # Create result
    result = movies.iloc[
        movie_indices_result
    ].copy()

    # Similarity percentage
    result["similarity"] = [
        score[1] * 100
        for score in similarity_scores
    ]

    return result


# =========================================================
# MOVIE SEARCH
# =========================================================

st.markdown(
    '<div class="section-title">🔎 Find a Movie</div>',
    unsafe_allow_html=True
)

movie_titles = sorted(
    movies["title"]
    .dropna()
    .astype(str)
    .unique()
)

selected_movie = st.selectbox(
    "Select a movie to get recommendations",
    movie_titles
)


# =========================================================
# SELECTED MOVIE
# =========================================================

selected_rows = movies[
    movies["title"] == selected_movie
]

if selected_rows.empty:

    st.error(
        "Selected movie could not be found."
    )

    st.stop()

selected_data = selected_rows.iloc[0]

selected_movie_id = selected_data["id"]

selected_poster = get_poster(
    selected_movie_id
)


# =========================================================
# SELECTED MOVIE SECTION
# =========================================================

st.markdown(
    '<div class="section-title">🎞️ Selected Movie</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(
    [1, 2]
)


# =========================================================
# SELECTED MOVIE POSTER
# =========================================================

with col1:

    if selected_poster:

        st.image(
            selected_poster,
            use_container_width=True
        )

    else:

        st.markdown("""
        <div style="
            height:300px;
            background:#191920;
            border-radius:10px;
            display:flex;
            align-items:center;
            justify-content:center;
            text-align:center;
            color:#777;
            font-size:20px;
        ">
            🎬<br>
            Poster unavailable
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# SELECTED MOVIE DETAILS
# =========================================================

with col2:

    st.markdown(
        f'<div class="movie-title">{selected_data["title"]}</div>',
        unsafe_allow_html=True
    )

    # Rating
    try:

        rating = float(
            selected_data["vote_average"]
        )

    except:

        rating = 0

    st.markdown(
        f'<div class="movie-meta">⭐ Rating: {rating:.1f}/10</div>',
        unsafe_allow_html=True
    )

    st.write("")

    # Genres
    genres = selected_data["genres"]

    genre_html = ""

    for genre in genres:

        genre_html += (
            f'<span class="genre-tag">{genre}</span>'
        )

    if genre_html:

        st.markdown(
            genre_html,
            unsafe_allow_html=True
        )

    st.write("")

    # Overview
    overview = selected_data["overview"]

    if not overview:
        overview = "No overview available."

    st.write(overview)

    st.write("")

    # Director
    director = selected_data["director"]

    if director:

        st.markdown(
            f"**Director:** {director}"
        )


# =========================================================
# GET RECOMMENDATIONS
# =========================================================

recommendations = recommend_movies(
    selected_movie,
    5
)


# =========================================================
# RECOMMENDED MOVIES
# =========================================================

st.markdown(
    '<div class="section-title">✨ Recommended For You</div>',
    unsafe_allow_html=True
)


if recommendations.empty:

    st.warning(
        "No recommendations found."
    )

else:

    columns = st.columns(5)

    for column, (_, movie) in zip(
        columns,
        recommendations.iterrows()
    ):

        with column:

            movie_id = movie["id"]

            poster = get_poster(
                movie_id
            )

            # Poster
            if poster:

                st.image(
                    poster,
                    use_container_width=True
                )

            else:

                st.markdown("""
                <div style="
                    height:260px;
                    background:#191920;
                    border-radius:10px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    text-align:center;
                    color:#777;
                ">
                    🎬<br>
                    Poster unavailable
                </div>
                """, unsafe_allow_html=True)

            # Title
            st.markdown(
                f'<div class="movie-title">{movie["title"]}</div>',
                unsafe_allow_html=True
            )

            # Rating
            try:

                rating = float(
                    movie["vote_average"]
                )

            except:

                rating = 0

            # Similarity
            similarity_score = float(
                movie["similarity"]
            )

            st.markdown(
                f'<div class="movie-meta">⭐ {rating:.1f}/10</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="score">AI Match: {similarity_score:.1f}%</div>',
                unsafe_allow_html=True
            )

            # Progress bar
            st.markdown(
                f"""
                <div class="progress-container">
                    <div class="progress-bar"
                         style="width:{min(similarity_score, 100)}%;">
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# DETAILED RECOMMENDATIONS
# =========================================================

st.markdown(
    '<div class="section-title">📋 Recommendation Details</div>',
    unsafe_allow_html=True
)


for _, movie in recommendations.iterrows():

    col1, col2 = st.columns(
        [1, 4]
    )

    # -----------------------------------------------------
    # Poster
    # -----------------------------------------------------

    with col1:

        poster = get_poster(
            movie["id"]
        )

        if poster:

            st.image(
                poster,
                use_container_width=True
            )

        else:

            st.markdown("""
            <div style="
                height:220px;
                background:#191920;
                border-radius:10px;
                display:flex;
                align-items:center;
                justify-content:center;
                text-align:center;
                color:#777;
            ">
                🎬<br>
                Poster unavailable
            </div>
            """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # Details
    # -----------------------------------------------------

    with col2:

        st.markdown(
            f'<div class="movie-title">{movie["title"]}</div>',
            unsafe_allow_html=True
        )

        # Rating
        try:

            rating = float(
                movie["vote_average"]
            )

        except:

            rating = 0

        # Similarity
        similarity_score = float(
            movie["similarity"]
        )

        st.markdown(
            f"⭐ **Rating:** {rating:.1f}/10"
        )

        st.markdown(
            f"🤖 **AI Similarity:** {similarity_score:.1f}%"
        )

        # Progress
        st.markdown(
            f"""
            <div class="progress-container">
                <div class="progress-bar"
                     style="width:{min(similarity_score, 100)}%;">
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Genres
        genres = movie["genres"]

        genre_html = ""

        for genre in genres:

            genre_html += (
                f'<span class="genre-tag">{genre}</span>'
            )

        if genre_html:

            st.markdown(
                genre_html,
                unsafe_allow_html=True
            )

        # Overview
        overview = movie["overview"]

        if not overview:
            overview = "No overview available."

        st.write(overview)

    st.divider()


# =========================================================
# HOW AI WORKS
# =========================================================

st.markdown(
    '<div class="section-title">🧠 How AI Works</div>',
    unsafe_allow_html=True
)

with st.expander(
    "Understand the recommendation process"
):

    st.markdown("""
    ### Step 1 — Movie Data

    The system uses the TMDB 5000 Movie Dataset.

    The dataset contains information such as movie title,
    overview, genres, keywords, cast and director.

    ### Step 2 — Feature Extraction

    Important movie information is combined into a single
    text representation called tags.

    ### Step 3 — TF-IDF

    TF-IDF converts the movie text into numerical vectors.

    Words that help distinguish movies receive greater
    importance in the representation.

    ### Step 4 — Cosine Similarity

    Cosine similarity compares the numerical representation
    of movies.

    Movies with similar textual features receive higher
    similarity values.

    ### Step 5 — Recommendation

    When a user selects a movie, the system compares it
    with all other movies.

    The five movies with the highest similarity scores
    are displayed as recommendations.
    """)


# =========================================================
# TECHNOLOGIES USED
# =========================================================

st.markdown(
    '<div class="section-title">⚙️ Technologies Used</div>',
    unsafe_allow_html=True
)

tech1, tech2, tech3, tech4 = st.columns(4)


with tech1:

    st.markdown("""
    <div class="feature-card">
        <h3>🐍 Python</h3>
        <p>
        Programming language used to build the
        recommendation system.
        </p>
    </div>
    """, unsafe_allow_html=True)


with tech2:

    st.markdown("""
    <div class="feature-card">
        <h3>📊 Pandas</h3>
        <p>
        Used for loading, merging and processing
        movie dataset information.
        </p>
    </div>
    """, unsafe_allow_html=True)


with tech3:

    st.markdown("""
    <div class="feature-card">
        <h3>🤖 Scikit-learn</h3>
        <p>
        Used for TF-IDF feature extraction and
        cosine similarity calculation.
        </p>
    </div>
    """, unsafe_allow_html=True)


with tech4:

    st.markdown("""
    <div class="feature-card">
        <h3>🌐 Streamlit</h3>
        <p>
        Used to create the interactive web
        application interface.
        </p>
    </div>
    """, unsafe_allow_html=True)