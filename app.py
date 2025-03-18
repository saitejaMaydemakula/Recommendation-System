import streamlit as st
import pandas as pd
import numpy as np
import heapq
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
import requests


# Function to load vectors
def load_vectors():
    return load_npz("vectors.npz")


vectors = load_vectors()


# Function to create or load SQLite database
def setup_database():
    conn = sqlite3.connect("movies.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS movies (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()

    # Load data if the table is empty
    cursor.execute("SELECT COUNT(*) FROM movies")
    count = cursor.fetchone()[0]
    if count == 0:
        df = pd.read_csv("movies_updated.csv", usecols=["name"])
        df.to_sql("movies", conn, if_exists="replace", index=False)

    conn.close()


setup_database()


# Function to search movies dynamically
def search_movies(query):
    conn = sqlite3.connect("movies.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM movies WHERE name LIKE ?", ('%' + query + '%',))
    results = cursor.fetchall()
    conn.close()
    return [r[0] for r in results]


# Fetch movie poster
def fetch_movie_poster(movie_name, api_key):
    movie_name = movie_name.strip().replace(" ", "+")
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_name}&language=en-US"
    response = requests.get(search_url, timeout=5)

    if response.status_code == 200:
        results = response.json().get("results", [])
        if results:
            movie_id = results[0]["id"]
            movie_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
            movie_response = requests.get(movie_url, timeout=5)

            if movie_response.status_code == 200:
                poster_path = movie_response.json().get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w780{poster_path}"

    return None


# Recommendation function
def recommend(movie, k=10):
    conn = sqlite3.connect("movies.db")
    df = pd.read_sql("SELECT name FROM movies", conn)
    conn.close()

    movie_index = df[df["name"] == movie].index[0]
    distances = cosine_similarity(vectors[movie_index], vectors).flatten()
    top_k_indices = heapq.nlargest(k + 1, range(len(distances)), key=lambda i: distances[i])
    top_k_indices = [i for i in top_k_indices if i != movie_index][:k]

    columns = st.columns(3)
    j = 0
    for i in top_k_indices:
        name = df.iloc[i, 0]
        url = fetch_movie_poster(name, "3bb01f22c75c33340bfe0c7dec4be139")
        if url:
            with columns[j % 3]:
                st.image(url, caption=name, width=200)
                j = (j + 1) % 3


# Streamlit UI
st.title("Movies Recommender System")

user_input = st.text_input("Enter the movie name")

if user_input:
    movie_options = search_movies(user_input)
    if movie_options:
        option = st.selectbox("Select a movie", movie_options)
    else:
        st.warning("No matching movies found. Try a different keyword.")
        option = None
else:
    option = None

if option and st.button("Recommend", type="primary"):
    with st.spinner("Fetching movie posters..."):
        recommend(option)
