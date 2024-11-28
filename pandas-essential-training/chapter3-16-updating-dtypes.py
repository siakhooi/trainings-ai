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
  df.Sport = df.Sport.str.lower()
  df.Discipline = df.Discipline.str.lower()
  df.Event = df.Event.str.lower()
  df.NOC = df.NOC.str.upper()
  df["Event Gender"] = df["Event Gender"].astype("category")
  df.loc[24676, "Gender"] = "Women"
  return df

def preprocess_2008(filename="olympics_2008.csv"):
  print(f"Preprocessing {filename} ...\n")
  df = pd.read_csv(filename)
  df.columns = ['City', 'Year', 'Sport', 'Discipline', 'Athlete Name', 'NOC',
       'Gender', 'Event', 'Event Gender', 'Medal', 'Result']
  df = df.drop("Result", axis=1)
  df.City = df.City.fillna(value="Beijing")
  df.Year = df.Year.fillna(value=2008)
  df = df.dropna(subset=['Sport', 'Discipline', 'Athlete Name', 'NOC', 'Gender',
       'Event', 'Event Gender', 'Medal'], how="all")
  df = df.drop_duplicates()
  df.Sport = df.Sport.str.lower()
  df.Discipline = df.Discipline.str.lower()
  df.Event = df.Event.str.lower()
  df.NOC = df.NOC.str.upper()
  df.Medal = df.Medal.str.capitalize()
  df.City = df.City.astype("string")
  df.Year = df.Year.astype(int)
  df.Sport = df.Sport.astype("string")
  df.Discipline = df.Discipline.astype("string")
  df["Athlete Name"] = df["Athlete Name"].astype("string")
  df.NOC = df.NOC.astype("string")
  df.Gender = df.Gender.astype("category")
  df.Event = df.Event.astype("string")
  df['Event Gender'] = df['Event Gender'].astype("category")
  medal_order = ["Bronze", "Silver", "Gold"]
  df.Medal = pd.Categorical(df.Medal, categories=medal_order, ordered=True)

  return df

oo = preprocess()
nw = preprocess_2008()
nw.sample(3)
