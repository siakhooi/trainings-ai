import pandas as pd

filename = "olympics_1896_2004.csv"

oo = pd.read_csv(filename, skiprows=5)

# print(oo.Medal.unique())
# oo.Medal = oo.Medal.astype("category")
# oo.dtypes

oo.Gender.unique()
oo['Event Gender'].unique()
oo.Gender = oo.Gender.astype("category")
oo['Event Gender'] = oo['Event Gender'].astype("category")

print(oo.dtypes)

medal_order = ["Bronze", "Silver", "Gold"]
oo.Medal = pd.Categorical(oo.Medal, categories=medal_order, ordered=True)

print(oo.dtypes)

print(oo.sort_values(by=["Year", "Event", "Medal"], ascending=[True, True, False]).head(7))
