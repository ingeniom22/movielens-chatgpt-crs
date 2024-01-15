import pandas as pd
import random
import json

df = pd.read_csv("lkpp_dataset_clean.csv", sep=";")
print(df.columns)

ppk_df = pd.DataFrame()
ppk_df["ppk_id"] = df["ppk_id"]
ppk_df.drop_duplicates(inplace=True)

ppk_df["kode_provinsi"] = [random.randint(1, 34) for _ in range(len(ppk_df))]
ppk_df["ippd"] = [random.choice(["A", "B", "C", "D"]) for _ in range(len(ppk_df))]
ppk_df["unit_kerja"] = [
    random.choice(["Kementerian", "Pemda", "Pemprov", "Pemkot"])
    for _ in range(len(ppk_df))
]

ppk_df["level_jabatan"] = [random.randint(1, 5) for _ in range(len(ppk_df))]

df = pd.merge(df, ppk_df, on="ppk_id")

print(ppk_df.head())
print(ppk_df.shape)

print(df.head())

df.to_csv("dummy_lkpp.csv", index=False)


company_df = pd.read_json("company_identity.jsonl", lines=True)[
    ["company_id", "penyedia"]
]

penyedia_id_dict = dict(zip(company_df["company_id"], company_df["penyedia"]))
with open('penyedia_id_mappings.json', 'w') as json_file:
    json.dump(penyedia_id_dict, json_file)
