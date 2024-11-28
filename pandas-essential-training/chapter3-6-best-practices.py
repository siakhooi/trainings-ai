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
oo.sample(3)

import re

search_string = "excel"

print([func for func in dir(pd) if re.search(rf"{search_string}", func, re.IGNORECASE)])

#assert

assert(oo[(oo.Year < 1896) & (oo.Year > 2004)].shape[0] == 0)

#isin()

years_of_interest = [1972, 1980, 1984, 1992, 2000, 2004]

print(oo[oo.Year.isin(years_of_interest)])
print(oo[~oo.Year.isin(years_of_interest)])

oo.info()

