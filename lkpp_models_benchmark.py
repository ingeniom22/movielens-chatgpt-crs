import pandas as pd
import tensorflow as tf
from libreco.algorithms import (
    DeepFM,
    GraphSage,
    PinSage,
    WideDeep,
    YouTubeRanking,
)
from libreco.data import DatasetFeat, split_by_ratio_chrono
from libreco.evaluation import evaluate

# from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "lkpp_model"
tf.compat.v1.reset_default_graph()

# data = pd.read_csv("sample_movielens_merged.csv", sep=",", header=0)
data = pd.read_csv("data/dummy_lkpp.csv")

print(data.columns)

data.rename(
    {
        "ppk_id": "user",
        "company_id": "item",
        "Rescalling": "label",
        "timestamp": "time",
    },
    axis="columns",
    inplace=True,
)

columns = [
    "user",
    "item",
    "time",
    "rt_akurasi",
    "rt_layanan",
    "rt_kirim",
    "category",
    "Sum",
    "label",
    "kode_provinsi",
    "ippd",
    "unit_kerja",
    "level_jabatan",
]

data = data[columns]

train, test = split_by_ratio_chrono(data, test_size=0.2)

# sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
sparse_col = [
    "category",
    "kode_provinsi",
    "ippd",
    "unit_kerja",
]

dense_col = ["level_jabatan"]

user_col = ["kode_provinsi", "ippd", "unit_kerja", "level_jabatan"]
# item_col = ["genre1", "genre2", "genre3"]
item_col = ["category"]

metrics = [
    "loss",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "map",
    "ndcg",
]


def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 30, name, "=" * 30)


train_data, data_info = DatasetFeat.build_trainset(
    train, user_col, item_col, sparse_col, dense_col, shuffle=False
)
test_data = DatasetFeat.build_testset(test, shuffle=False)
print(data_info)

reset_state("DeepFM")
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
    metrics=metrics,
)

reset_state("GraphSage")
graphsage = GraphSage(
    "ranking",
    data_info,
    loss_type="max_margin",
    paradigm="i2i",
    embed_size=16,
    n_epochs=2,
    lr=3e-4,
    lr_decay=False,
    reg=None,
    batch_size=2048,
    num_neg=1,
    dropout_rate=0.0,
    num_layers=1,
    num_neighbors=10,
    num_walks=10,
    sample_walk_len=5,
    margin=1.0,
    sampler="random",
    start_node="random",
    focus_start=False,
    seed=42,
)
graphsage.fit(
    train_data,
    neg_sampling=True,
    verbose=2,
    shuffle=True,
    eval_data=test_data,
    metrics=metrics,
)

reset_state("PinSage")
pinsage = PinSage(
    "ranking",
    data_info,
    loss_type="cross_entropy",
    paradigm="u2i",
    embed_size=16,
    n_epochs=2,
    lr=3e-4,
    lr_decay=False,
    reg=None,
    batch_size=2048,
    num_neg=1,
    dropout_rate=0.0,
    remove_edges=False,
    num_layers=1,
    num_neighbors=10,
    num_walks=10,
    neighbor_walk_len=2,
    sample_walk_len=5,
    termination_prob=0.5,
    margin=1.0,
    sampler="random",
    start_node="random",
    focus_start=False,
    seed=42,
)
pinsage.fit(
    train_data,
    neg_sampling=True,
    verbose=2,
    shuffle=True,
    eval_data=test_data,
    metrics=metrics,
)

# # save data_info, specify model save folder
data_info.save(path=MODEL_PATH, model_name="pinsage_model_lkpp")
# # set manual=True will use `numpy` to save model
# # set manual=False will use `tf.train.Saver` to save model
# # set inference=True will only save the necessary variables for prediction and recommendation
pinsage.save(
    path=MODEL_PATH, model_name="pinsage_model_lkpp", manual=True, inference_only=True
)

reset_state("WideDeep")
wd = WideDeep(
    "ranking",
    data_info,
    loss_type="cross_entropy",
    embed_size=16,
    n_epochs=2,
    lr={"wide": 0.01, "deep": 1e-4},
    lr_decay=False,
    reg=None,
    batch_size=2048,
    num_neg=1,
    use_bn=False,
    dropout_rate=None,
    hidden_units=(128, 64, 32),
    tf_sess_config=None,
)
wd.fit(
    train_data,
    neg_sampling=True,
    verbose=2,
    shuffle=True,
    eval_data=test_data,
    metrics=metrics,
)

reset_state("YoutubeRanking")
ytb_ranking = YouTubeRanking(
    "ranking",
    data_info,
    loss_type="cross_entropy",
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
ytb_ranking.fit(
    train_data,
    neg_sampling=True,
    verbose=2,
    shuffle=True,
    eval_data=test_data,
    metrics=metrics,
)


models = [deepfm, graphsage, pinsage, wd, ytb_ranking]
eval_results_list = []
for model in models:
    eval_result = evaluate(
        model=model,
        data=test_data,
        neg_sampling=True,
        eval_batch_size=8192,
        k=10,
        metrics=metrics,
        # sample_user_num=2048,
        seed=2222,
    )
    eval_result["model_name"] = str(model)
    eval_results_list.append(eval_result)
    print(eval_result)

# Membuat dataframe dari list hasil evaluasi
df_eval_results = pd.DataFrame(eval_results_list)

# Menyimpan dataframe ke dalam file CSV
df_eval_results.to_csv("eval_results/eval_results_lkpp.csv", index=False)
