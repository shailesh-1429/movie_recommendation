import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page configuration
st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    layout="wide"
)

# 2. Load saved model artifacts
@st.cache_resource
def load_model():
    similarity = joblib.load("item_similarity.pkl")
    train_matrix = joblib.load("train_matrix.pkl")
    global_mean = joblib.load("global_mean.pkl")
    movies = pd.read_csv("movie_info.csv")

    return similarity, train_matrix, global_mean, movies


try:
    item_similarity_df, train_matrix, global_mean, movie_info = load_model()
except FileNotFoundError:
    st.error(
        "Model files not found. Place all four saved artifacts "
        "in the same folder as app.py."
    )
    st.stop()


# 3. Prepare movie information

movie_info["movie_id"] = movie_info["movie_id"].astype(int)

movie_titles = (
    movie_info.drop_duplicates("movie_id")
    .set_index("movie_id")["title"]
)

# 4. Recommendation functions
def find_similar_movies(movie_id, n=5):
    scores = item_similarity_df[movie_id].drop(movie_id).sort_values(ascending=False).head(n)
    result = pd.DataFrame({
        "movie_id": scores.index,
        "Similarity score": scores.values
    })

    result["Movie title"] = result["movie_id"].map(movie_titles)

    return result[["Movie title", "Similarity score"]]


def predict_rating(user_id, movie_id):
    if user_id not in train_matrix.index:
        return global_mean

    if movie_id not in item_similarity_df.index:
        return global_mean

    user_ratings = train_matrix.loc[user_id].dropna()

    similarities = item_similarity_df.loc[
        movie_id, user_ratings.index
    ]

    # Use only movies with positive similarity
    similarities = similarities[similarities > 0]

    if similarities.empty:
        return float(global_mean)

    ratings = user_ratings.loc[similarities.index]

    prediction = (
        (similarities * ratings).sum()
        / similarities.abs().sum()
    )

    return prediction


def recommend_movies(user_id, n=10):
    if user_id not in train_matrix.index:
        return pd.DataFrame()

    watched_movies = set(
        train_matrix.loc[user_id].dropna().index
    )

    unseen_movies = [
        movie_id
        for movie_id in train_matrix.columns
        if movie_id not in watched_movies
        and movie_id in item_similarity_df.index
    ]

    predictions = [
        (int(movie_id), predict_rating(user_id, movie_id))
        for movie_id in unseen_movies
    ]

    predictions.sort(key=lambda x: x[1], reverse=True)

    result = pd.DataFrame(
        predictions[:n],
        columns=["movie_id", "Predicted rating"]
    )

    result["Movie title"] = result["movie_id"].map(movie_titles)

    return result[
        ["Movie title", "Predicted rating"]
    ].reset_index(drop=True)


# 5. Design the application UI
st.title("CineMatch")
st.write(
    "Explore movies with similar rating patterns or get "
    "recommendations based on a user's rating history."
)
st.divider()

similar_tab, personal_tab = st.tabs([
    "Find Similar Movies",
    "For You"
])

# 6. Similar movie recommendations
with similar_tab:
    st.header("Find movies like your favourite")

    available_movies = movie_info.sort_values("title")

    selected_title = st.selectbox(
        "Choose a movie",
        available_movies["title"].tolist(),
        key="similar_movie"
    )

    if st.button("Find Similar Movies", type="primary"):
        selected_row = available_movies[
            available_movies["title"] == selected_title
        ].iloc[0]

        selected_id = int(selected_row["movie_id"])

        st.markdown(f"### Because you selected: {selected_title}")

        results = find_similar_movies(selected_id, n=5)

        if results.empty:
            st.info("No similar movies were found.")
        else:
            results["Similarity score"] = results[
                "Similarity score"
            ].round(3)

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )

# 7. Personalized recommendations
with personal_tab:
    st.header("Your personalized movie list")

    user_ids = sorted(int(user) for user in train_matrix.index)

    selected_user = st.selectbox(
        "Select a MovieLens user",
        user_ids,
        key="selected_user"
    )

    st.caption(
        "Choose an existing MovieLens user to see recommendations "
        "based on their recorded ratings."
    )

    if st.button("Recommend Movies", type="primary"):
        results = recommend_movies(selected_user, n=10)

        if results.empty:
            st.info("No recommendations are available for this user.")
        else:
            results["Predicted rating"] = results[
                "Predicted rating"
            ].round(2)

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )

st.divider()