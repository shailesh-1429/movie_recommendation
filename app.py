
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

# ---------------- PREMIUM BLACK & GOLD CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    .stApp {
        background: #0B0B0D;
        color: #F5F2EA;
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 11, 13, 0.95);
    }

    [data-testid="stSidebar"] {
        background: #111114;
        border-right: 1px solid #302719;
    }

    .brand {
        color: #D9B66D;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 4px;
        margin-bottom: 8px;
    }

    .hero {
        background: linear-gradient(120deg, #1B1710, #111114 65%);
        border: 1px solid #49391E;
        border-radius: 20px;
        padding: 38px 34px;
        margin: 12px 0 28px 0;
    }

    .hero h1 {
        font-family: 'Playfair Display', serif;
        color: #F3D18B;
        font-size: clamp(32px, 4vw, 48px);
        margin: 0 0 12px 0;
    }

    .hero p {
        color: #C0B9AA;
        font-size: 16px;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .section-title {
        color: #E8C77E;
        font-family: 'Playfair Display', serif;
        font-size: 27px;
        margin: 22px 0 6px 0;
    }

    .muted {
        color: #A7A19A;
        font-size: 14px;
        margin-bottom: 20px;
    }

    .movie-card {
        background: linear-gradient(145deg, #19191D, #111114);
        border: 1px solid #35302A;
        border-radius: 15px;
        padding: 20px 17px;
        min-height: 160px;
        margin-bottom: 12px;
        transition: border-color 0.2s ease;
    }

    .movie-card:hover {
        border-color: #D9B66D;
    }

    .movie-number {
        color: #D9B66D;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
    }

    .movie-title {
        color: #F7F1E5;
        font-size: 17px;
        font-weight: 700;
        line-height: 1.5;
        margin: 12px 0;
        overflow-wrap: anywhere;
    }

    .movie-meta {
        color: #AAA39A;
        font-size: 12px;
    }

    .stButton > button {
        background: linear-gradient(100deg, #E5C47C, #C49A48);
        color: #17130B;
        border: none;
        border-radius: 9px;
        font-weight: 700;
        padding: 11px 18px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #F4DDA7;
        color: #17130B;
        border: none;
    }

    .stTextInput input, .stNumberInput input,
    .stSelectbox [data-baseweb="select"] {
        background: #19191D;
        color: #F5F2EA;
        border-color: #49402F;
        border-radius: 9px;
    }

    div[data-baseweb="select"] * {
        color: #F5F2EA;
    }

    hr {
        border-color: #302719;
    }

    .footer {
        text-align: center;
        color: #817A70;
        font-size: 12px;
        padding: 25px 0 8px 0;
    }

    .notice {
        background: #19170F;
        border: 1px solid #554421;
        border-radius: 10px;
        padding: 14px 16px;
        color: #DCC48E;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL FILES ----------------
@st.cache_resource
def load_model():
    similarity = joblib.load(BASE_DIR / "item_similarity.pkl")
    train_matrix = joblib.load(BASE_DIR / "train_matrix.pkl")
    global_mean = joblib.load(BASE_DIR / "global_mean.pkl")
    movie_info = pd.read_csv(BASE_DIR / "movie_info.csv")

    required = {"movie_id", "title"}
    if not required.issubset(movie_info.columns):
        raise ValueError(
            "movie_info.csv must contain movie_id and title columns."
        )

    if not isinstance(similarity, pd.DataFrame):
        similarity = pd.DataFrame(similarity)

    if not isinstance(train_matrix, pd.DataFrame):
        train_matrix = pd.DataFrame(train_matrix)

    movie_info = movie_info.dropna(subset=["movie_id", "title"]).copy()

    return similarity, train_matrix, global_mean, movie_info


# ---------------- RECOMMENDATION FUNCTIONS ----------------
def find_similar_movies(movie_id, n=5):
    """Find movies with the highest positive similarity."""
    if movie_id not in similarity.index:
        return pd.DataFrame()

    scores = similarity.loc[movie_id]

    scores = pd.to_numeric(scores, errors="coerce").dropna()
    scores = scores.drop(labels=[movie_id], errors="ignore")
    scores = scores[scores > 0].sort_values(ascending=False).head(n)

    result = pd.DataFrame({
        "movie_id": scores.index,
        "similarity": scores.values
    })

    return result.merge(movie_info, on="movie_id", how="inner")


def predict_rating(user_id, movie_id):
    """Predict a rating using positive item similarities."""
    try:
        if user_id not in train_matrix.index:
            return float(global_mean)

        user_ratings = train_matrix.loc[user_id].dropna()

        if movie_id not in similarity.index:
            return float(global_mean)

        common_movies = [
            mid for mid in user_ratings.index
            if mid in similarity.columns
            and mid != movie_id
            and user_ratings[mid] > 0
        ]

        if not common_movies:
            return float(global_mean)

        similarities = pd.to_numeric(
            similarity.loc[movie_id, common_movies],
            errors="coerce"
        )

        ratings = pd.to_numeric(
            user_ratings.loc[common_movies],
            errors="coerce"
        )

        valid = (
            similarities.notna()
            & ratings.notna()
            & (similarities > 0)
        )

        similarities = similarities[valid]
        ratings = ratings[valid]

        if similarities.empty or similarities.sum() == 0:
            return float(global_mean)

        predicted = np.dot(similarities, ratings) / similarities.sum()

        return float(predicted)

    except (KeyError, TypeError, ValueError):
        return float(global_mean)


def recommend_movies(user_id, n=10):
    """Recommend highly predicted movies the user has not rated."""
    if user_id not in train_matrix.index:
        return pd.DataFrame()

    user_ratings = train_matrix.loc[user_id].dropna()

    watched_movies = set(user_ratings.index)

    candidates = [
        mid for mid in similarity.index
        if mid not in watched_movies
    ]

    predictions = []

    for movie_id in candidates:
        rating = predict_rating(user_id, movie_id)
        predictions.append((movie_id, rating))

    if not predictions:
        return pd.DataFrame()

    predictions.sort(key=lambda item: item[1], reverse=True)

    result = pd.DataFrame(
        predictions[:n],
        columns=["movie_id", "predicted_rating"]
    )

    return result.merge(movie_info, on="movie_id", how="inner")


# ---------------- HEADER ----------------
st.markdown(
    '<div class="brand">CINEMATCH <span style="color:#777">'
    '/ YOUR PERSONAL MOVIE GUIDE</span></div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero">
    <h1>Every great story<br>starts with a recommendation.</h1>
    <p>Discover movies that match your taste.
    Explore similar titles and find your next favourite film.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
try:
    similarity, train_matrix, global_mean, movie_info = load_model()
except Exception as e:
    st.error("Movie recommendation files could not be loaded.")
    st.info(
        "Check that item_similarity.pkl, train_matrix.pkl, "
        "global_mean.pkl and movie_info.csv exist beside app.py."
    )
    with st.expander("Technical details"):
        st.code(str(e))
    st.stop()


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown(
        '<div class="brand" style="font-size:20px;letter-spacing:2px">'
        '🎬 CINEMATCH</div>',
        unsafe_allow_html=True
    )

    st.caption("YOUR MOVIE DISCOVERY SPACE")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Home", "Similar Movies", "About"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("**YOUR COLLECTION**")
    st.caption(f"Movies available: {len(movie_info):,}")
    st.caption(f"Users in model: {len(train_matrix.index):,}")

    st.markdown("---")
    st.caption("Premium discovery. Powered by recommendations.")


# ---------------- MOVIE CARD HELPER ----------------
def display_movie_cards(df, score_column=None, limit=10):
    if df.empty:
        st.warning(
            "No movies found. Try another movie or check the model data."
        )
        return

    df = df.head(limit).reset_index(drop=True)

    for start in range(0, len(df), 3):
        cols = st.columns(3)

        for col, (_, movie) in zip(
            cols, df.iloc[start:start + 3].iterrows()
        ):
            with col:
                title = str(movie["title"])
                movie_id = movie["movie_id"]

                if score_column and score_column in movie.index:
                    score = movie[score_column]
                    if score_column == "similarity":
                        meta = f"Similarity: {score:.0%}"
                    else:
                        meta = f"Predicted rating: {score:.2f}"
                else:
                    meta = "Recommended for you"

                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-number">CINEMATCH
                        &nbsp; / &nbsp; {start + 1 + list(
                            df.iloc[start:start + 3].index
                        ).index(movie.name):02d}</div>
                        <div class="movie-title">{title}</div>
                        <div class="movie-meta">{meta}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ---------------- HOME PAGE ----------------
if page == "Home":
    st.markdown(
        '<div class="section-title">Find your next favourite</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="muted">Enter your user ID to get personalised '
        'movie recommendations.</div>',
        unsafe_allow_html=True
    )

    with st.form("recommendation_form"):
        user_id_text = st.text_input(
            "User ID",
            placeholder="Enter a user ID from your dataset"
        )

        number_of_movies = st.select_slider(
            "Number of recommendations",
            options=[3, 6, 9, 12],
            value=6
        )

        submitted = st.form_submit_button(
            "✦ Discover Movies",
            use_container_width=True
        )

    if submitted:
        if not user_id_text.strip():
            st.warning("Please enter a user ID.")
        else:
            # Match text input with numeric or string index IDs.
            matching_id = None

            for existing_id in train_matrix.index:
                if str(existing_id) == user_id_text.strip():
                    matching_id = existing_id
                    break

            if matching_id is None:
                st.warning(
                    "User ID not found in the training data. "
                    "Please use a valid ID from your dataset."
                )
            else:
                with st.spinner("Finding movies for you..."):
                    recommendations = recommend_movies(
                        matching_id, n=number_of_movies
                    )

                st.markdown("---")
                st.markdown(
                    '<div class="section-title">Picked for you ✨</div>',
                    unsafe_allow_html=True
                )

                display_movie_cards(
                    recommendations,
                    score_column="predicted_rating",
                    limit=number_of_movies
                )


# ---------------- SIMILAR MOVIES PAGE ----------------
elif page == "Similar Movies":
    st.markdown(
        '<div class="section-title">Explore similar movies</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="muted">Choose a movie to discover titles '
        'with similar patterns in the model.</div>',
        unsafe_allow_html=True
    )

    movie_options = movie_info.drop_duplicates(
        subset=["movie_id"]
    ).sort_values("title")

    selected_title = st.selectbox(
        "Choose a movie",
        movie_options["title"].astype(str).tolist()
    )

    if st.button("✦ Find Similar Movies", use_container_width=True):
        selected_row = movie_options[
            movie_options["title"].astype(str) == selected_title
        ].iloc[0]

        selected_id = selected_row["movie_id"]

        similar = find_similar_movies(selected_id, n=9)

        st.markdown("---")
        st.markdown(
            f'<div class="section-title">Because you selected '
            f'{selected_title}</div>',
            unsafe_allow_html=True
        )

        display_movie_cards(
            similar,
            score_column="similarity",
            limit=9
        )


# ---------------- ABOUT PAGE ----------------
elif page == "About":
    st.markdown(
        '<div class="section-title">About CineMatch</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="movie-card">
        <div class="movie-title">Movies. Taste. Discovery.</div>
        <p style="color:#C0B9AA;line-height:1.8">
        CineMatch is a movie recommendation application that uses
        item similarity and user ratings to help discover films.
        </p>
        <hr>
        <div class="movie-meta">
        Built with Streamlit, Python, Pandas, NumPy and Joblib.
        </div>
    </div>
    """, unsafe_allow_html=True)

    a, b, c = st.columns(3)

    a.metric("Movies", f"{len(movie_info):,}")
    b.metric("Users", f"{len(train_matrix.index):,}")
    c.metric("Model", "Item Similarity")


# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    '<div class="footer">CINEMATCH · MADE FOR MOVIE LOVERS'
    '<br>Discover your next story.</div>',
    unsafe_allow_html=True
)
