
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = {
    "movie": [
        "The Avengers",
        "Avengers: Endgame",
        "Spider-Man: No Way Home",
        "Jurassic World",
        "Avatar",
        "Avatar: The Way of Water",
        "The Lion King"
    ],
    "action":    [1, 1, 1, 1, 1, 1, 0],
    "romance":   [0, 0, 0, 0, 0, 0, 0],
    "sci_fi":    [1, 1, 1, 1, 1, 1, 0],
    "animation": [0, 0, 0, 0, 0, 0, 1]
}

df = pd.DataFrame(data)

features = df[["action", "romance", "sci_fi", "animation"]]
similarity = cosine_similarity(features)

def recommend(movie_name):
    if movie_name not in df["movie"].values:
        print("Movie not found!")
        return

    index = df[df["movie"] == movie_name].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nRecommended movies:")
    for i in scores[1:4]:
        print(df.iloc[i[0]]["movie"])

# Single input
movie = input("Enter movie name: ")
recommend(movie)

