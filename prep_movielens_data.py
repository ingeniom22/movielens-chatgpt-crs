import json
import pandas as pd

users_df = pd.read_csv(
    "ml-100k/u.user",
    sep="|",
    names=["user_id", "age", "gender", "occupation", "zip_code"],
)

ratings_df = pd.read_csv(
    "ml-100k/u.data",
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"],
)

genres_df = pd.read_csv(
    "ml-100k/u.data",
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"],
)

items_df = pd.read_csv(
    "ml-100k/u.item",
    sep="|",
    names=[
        "item_id",
        "movie_title",
        "release_date",
        "video_release_date",
        "IMDb_URL",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ],
    encoding="latin-1",
)

print(users_df.head())
print(ratings_df.head())
print(items_df.head())


merged_df = pd.merge(ratings_df, users_df, on="user_id")
merged_df = pd.merge(merged_df, items_df, on="item_id")


print(merged_df.columns)
print(merged_df.head())

merged_df.to_csv("ml-100k-merged.csv", index=False)

movie_id_mappings = items_df[["item_id", "movie_title"]]
movie_id_dict = dict(zip(movie_id_mappings["item_id"], movie_id_mappings["movie_title"]))
with open('movie_id_mappings.json', 'w') as json_file:
    json.dump(movie_id_dict, json_file)

print(movie_id_dict)


