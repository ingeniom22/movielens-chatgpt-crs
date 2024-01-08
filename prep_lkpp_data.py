import pandas as pd
import random

df = pd.read_csv("lkpp_dataset_clean.csv", sep=";")
print(df.columns)

ppk_df = pd.DataFrame()
ppk_df["ppk_id"] = df["ppk_id"]
ppk_df.drop_duplicates(inplace=True)

ppk_df["kode_provinsi"] = [random.randint(1, 34) for _ in range(len(ppk_df))]
ppk_df["ippd"] = [random.choice(['A', 'B', 'C', 'D']) for _ in range(len(ppk_df))]
# ppk_df["nama"] =  


print(ppk_df.head())
print(ppk_df.shape) 