import pandas as pd

def preprocess_2008(filename="olympics_2008.csv"):
  df = pd.read_csv(filename)
  df.columns = ['City', 'Year', 'Sport', 'Discipline', 'Athlete Name', 'NOC',
       'Gender', 'Event', 'Event Gender', 'Medal', 'Result']
  df = df.drop("Result", axis=1)
  return df

nw = preprocess_2008()
nw.sample(3)
print(nw.info())

print(nw.City.isna().sum())
print(nw[nw.City.isna()])

nw.City = nw.City.fillna(value="Beijing")
print(nw.City.unique())


nw.Year = nw.Year.fillna(value=2008)
print(nw.info())

print(nw.dtypes)

def preprocess_2008(filename="olympics_2008.csv"):
  df = pd.read_csv(filename)
  df.columns = ['City', 'Year', 'Sport', 'Discipline', 'Athlete Name', 'NOC',
       'Gender', 'Event', 'Event Gender', 'Medal', 'Result']
  df = df.drop("Result", axis=1)
  df.City = df.City.fillna(value="Beijing")
  df.Year = df.Year.fillna(value=2008)
  return df



