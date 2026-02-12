from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
movies = pd.read_csv("movies.csv")
movies['genres'] = movies['genres'].fillna('')

# TF-IDF on genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Title-based recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        return None  # return None if not found
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Genre-based search
def search_by_genre(genre):
    genre = genre.lower()
    results = movies[movies['genres'].str.lower().str.contains(genre)]
    if results.empty:
        return ["No movies found for this genre."]
    return results['title'].head(5).tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        query = request.form.get("movie")
        recs = get_recommendations(query)
        if recs is None:  # if not a title, try genre
            recs = search_by_genre(query)
        recommendations = recs
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)