import pandas as pd

filename = "olympics_1896_2004.csv"

oo = pd.read_csv(filename, skiprows=5)

# Rename

mapper = {"Athlete Name": "Athlete_Name", "Event Gender": "Event_Gender"}

oo = oo.rename(columns=mapper)
print(oo.sample(3))


column_names = ['Year', 'City', 'Sport', 'Discipline', 'Athlete_Name', 'NOC', 'Gender',
       'Event', 'Event_Gender', 'Medal', 'Position']
oo.columns = column_names
print(oo.sample(3))

oo = pd.read_csv(filename, skiprows=5, names=column_names, header=0)
print(oo.head())


# Remove (col: axis=1, row: axis=0)

oo = oo.drop('Position', axis=1)
print(oo.sample(3))

oo = (pd.read_csv(filename, skiprows=5)
      .drop('Position', axis=1)
)
print(oo.sample(3))


oo = oo.drop(['City', 'Sport'], axis=1)
print(oo.sample(3))
