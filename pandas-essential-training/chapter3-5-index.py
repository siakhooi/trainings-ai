import pandas as pd

def preprocess(filename = "olympics_1896_2004.csv"):
  """Preparing and transforming dataframe"""
  print(f"Preprocessing {filename} ...\n")
  ordered_medals = pd.api.types.CategoricalDtype(categories=["Bronze", "Silver", "Gold"], ordered=True)
  dtype_mapper = {"Year": "int64",
                "City": "string",
                "Sport": "string",
                "Discipline": "string",
                "Athlete Name": "string",
                "NOC": "string",
                "Gender": "category",
                "Event": "string",
                "Event_gender": "category",
                "Medal": ordered_medals}
  df = (pd.read_csv(filename, skiprows=5, dtype=dtype_mapper)
        .drop('Position', axis=1)
  )
  df["Event Gender"] = df["Event Gender"].astype("category")
  return df

oo = preprocess()
print(oo.head(5))

oo.index

oo.columns

oo[oo['Athlete Name'].str.contains("LEWIS, Carl")]
oo.loc[22719, "Event"]

oo = oo.set_index("Athlete Name")
print(oo.head(5))
oo.index
oo.loc["LEWIS, Carl", "Event"]
oo.head(3)
oo.columns
oo.shape
oo.head(3)
oo.loc["LEWIS, Carl", ["Year", "Event", "Medal"]]
oo.loc["LEWIS, Carl", :]
oo.loc["LEWIS, Carl"]


oo = oo.sort_index()
print(oo.head(3))

oo = oo.reset_index()
print(oo.head(3))


print(oo.iloc[0, 0])
oo.iloc[0, 5]
oo.iloc[0, :]
print(oo.iloc[0])
