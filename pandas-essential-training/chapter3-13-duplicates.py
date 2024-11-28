import pandas as pd

def preprocess_2008(filename="olympics_2008.csv"):
  df = pd.read_csv(filename)
  df.columns = ['City', 'Year', 'Sport', 'Discipline', 'Athlete Name', 'NOC',
       'Gender', 'Event', 'Event Gender', 'Medal', 'Result']
  df = df.drop("Result", axis=1)
  df.City = df.City.fillna(value="Beijing")
  df.Year = df.Year.fillna(value=2008)
  df = df.dropna(subset=['Sport', 'Discipline', 'Athlete Name', 'NOC', 'Gender',
       'Event', 'Event Gender', 'Medal'], how="all")
  return df

nw = preprocess_2008()
print(nw.sample(3))

print(nw.duplicated())
print(nw.duplicated().sum())

print(nw.loc[nw.duplicated(), :])

nw = nw.drop_duplicates()
print(nw.shape)




athlete_multiple_events = nw.duplicated(subset=['Athlete Name', 'NOC', 'Gender'])
print(athlete_multiple_events)

print(nw.loc[athlete_multiple_events, :])
print(nw.loc[athlete_multiple_events, :].sort_values("Athlete Name"))

print(nw.loc[nw["Athlete Name"] == "ZOU, Kai"])


def preprocess_2008(filename="olympics_2008.csv"):
  df = pd.read_csv(filename)
  df.columns = ['City', 'Year', 'Sport', 'Discipline', 'Athlete Name', 'NOC',
       'Gender', 'Event', 'Event Gender', 'Medal', 'Result']
  df = df.drop("Result", axis=1)
  df.City = df.City.fillna(value="Beijing")
  df.Year = df.Year.fillna(value=2008)
  df = df.dropna(subset=['Sport', 'Discipline', 'Athlete Name', 'NOC', 'Gender',
       'Event', 'Event Gender', 'Medal'], how="all")
  df = df.drop_duplicates()
  return df

nw = preprocess_2008()
print(nw.sample(3))

