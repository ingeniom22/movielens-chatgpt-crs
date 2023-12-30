import pandas as pd

import tensorflow as tf

from libreco.algorithms import DeepFM
from libreco.data import DataInfo, DatasetFeat, split_by_ratio_chrono
from libreco.evaluation import evaluate

MODEL_PATH = "model"

# data = pd.read_csv("sample_movielens_merged.csv", sep=",", header=0)
data = pd.read_csv("ml-100k-merged.csv")
data.rename(
    {
        "user_id": "user",
        "item_id": "item",
        "rating": "label",
        "timestamp": "time",
    },
    axis="columns",
    inplace=True,
)
# print(data.head())

train, test = split_by_ratio_chrono(data, test_size=0.2)

sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
sparse_col = [
    "gender",
    "occupation",
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
]

dense_col = ["age"]
user_col = ["gender", "age", "occupation"]
item_col = ["genre1", "genre2", "genre3"]
item_col = [
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
]


train_data, data_info = DatasetFeat.build_trainset(
    train, user_col, item_col, sparse_col, dense_col, shuffle=False
)
test_data = DatasetFeat.build_testset(test, shuffle=False)
print(data_info)

deepfm = DeepFM(
    "ranking",
    data_info,
    embed_size=16,
    n_epochs=2,
    lr=1e-4,
    lr_decay=False,
    reg=None,
    batch_size=2048,
    num_neg=1,
    use_bn=False,
    dropout_rate=None,
    hidden_units=(128, 64, 32),
    tf_sess_config=None,
)

deepfm.fit(
    train_data,
    neg_sampling=True,
    verbose=2,
    shuffle=True,
    eval_data=test_data,
    metrics=[
        "loss",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "map",
        "ndcg",
    ],
    eval_batch_size=8192,
    k=10,
    eval_user_num=2048,
)

print("prediction: ", deepfm.predict(user=2211, item=110))
print("recommendation: ", deepfm.recommend_user(user=896, n_rec=7))

# save data_info, specify model save folder
data_info.save(path=MODEL_PATH, model_name="deepfm_model")
# set manual=True will use `numpy` to save model
# set manual=False will use `tf.train.Saver` to save model
# set inference=True will only save the necessary variables for prediction and recommendation
deepfm.save(
    path=MODEL_PATH, model_name="deepfm_model", manual=True, inference_only=True
)
