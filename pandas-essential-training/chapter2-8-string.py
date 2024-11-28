import pandas as pd

filename = "olympics_1896_2004.csv"

oo = pd.read_csv(filename, skiprows=5)

oo["Athlete Name"].str.lower()

oo.Event = oo.Event.str.capitalize()
print(oo.Event.unique())


print(oo[oo["Athlete Name"].str.contains("LATYNINA")])

oo.City = oo.City.str.upper()
print(oo.City.unique())
