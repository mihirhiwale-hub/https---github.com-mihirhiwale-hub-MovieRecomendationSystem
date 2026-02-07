from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
movies = pd.read_csv("movies.csv")   # Make sure movies.csv is in the same folder
movies['genres'] = movies['genres'].fillna('')

# Step 1: Feature extraction (TF-IDF on genres)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Step 2: Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Helper function to get recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        return ["Movie not found in dataset."]
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        movie_name = request.form.get("movie")
        recommendations = get_recommendations(movie_name)
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)