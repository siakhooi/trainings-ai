import pandas as pd

filename = "olympics_1896_2004.csv"

oo = pd.read_csv(filename, skiprows=5)

oo = (pd.read_csv(filename, skiprows=5)
      .drop('Position', axis=1)
)
print(oo.tail(3))

print(oo[(oo.Year == 1896) | (oo.Year == 2004)])

first_men_100m = oo[(oo.Year == 1896) & (oo.Gender == 'Men') & (oo.Event == '100m')]
print(first_men_100m)

print(first_men_100m[["Year", "Athlete Name", "NOC", "Event", "Medal"]])


print(oo[(oo.Year == 1896) & (oo.Gender == 'Men') & (oo.Event == '100m')][["Year", "Athlete Name", "NOC", "Event", "Medal"]])

