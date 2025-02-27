import streamlit as st
import pandas as pd
import numpy as np
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
import requests


def load_data():
    return load_npz('vectors.npz'), pd.read_csv('movies_updated.csv', usecols=['name'])

vectors, movies = load_data()
apikey = "3bb01f22c75c33340bfe0c7dec4be139"

def fetch_movie_poster(movie_name, api_key):
    movie_name = movie_name.strip().replace(' ', '+')
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_name}&language=en-US"
    response = requests.get(search_url, timeout=7)
    if response.status_code == 200:
        results = response.json().get('results', [])
        if results:
            movie_id = results[0]['id']
            movie_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
            movie_response = requests.get(movie_url, timeout=7)
            if movie_response.status_code == 200:
                poster_path = movie_response.json().get('poster_path')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w780{poster_path}"
    return None

def recommend(movie, k=10):
    movie_index = movies[movies['name'] == movie].index[0]
    distances = cosine_similarity(vectors[movie_index], vectors).flatten()
    top_k_indices = heapq.nlargest(k + 1, range(len(distances)), key=lambda i: distances[i])
    top_k_indices = [i for i in top_k_indices if i != movie_index][:k]
    columns=st.columns(3)
    j=0
    for i in top_k_indices:
        name = movies.iloc[i, 0]
        url = fetch_movie_poster(name, apikey)
        if url:
            with columns[j%3]:
                st.image(url,caption=name,width=200)
                j=(j+1)%3

st.title('Movies Recommender System')
option = st.selectbox("Enter The movie name", movies['name'].values)

if st.button("Recommend", type="primary") and option:
    with st.spinner("Fetching movie posters..."):
        recommend(option)
