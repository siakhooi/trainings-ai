import pandas as pd

filename = "olympics_1896_2004.csv"

oo = pd.read_csv(filename, skiprows=5)

print(type(oo))

print(oo["Discipline"])

print(oo.Discipline)

print(type(oo["Discipline"]))

print(oo.columns)

print(oo.Year.unique())

print(oo.Year.value_counts())

print(oo.Year.value_counts(normalize=True))

