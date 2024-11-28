import pandas as pd

filename = "olympics_1896_2004.csv"

oo = pd.read_csv(filename, skiprows=5)

print(oo.Year.unique())

print(oo.Medal.unique())

print(oo.Medal.value_counts())

print(oo.NOC.unique())
