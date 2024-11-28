import pandas as pd

filename = "olympics_1896_2004.csv"

oo = pd.read_csv(filename, skiprows=5)

oo["Athlete Name"].sort_values()

oo.sort_values("Athlete Name")
print(oo.head(3))


oo.sort_values("Athlete Name", ascending=False)

print(oo.sort_values("Athlete Name", ascending=False).head(25))


oo.sort_values(by=['Year','Athlete Name'])

oo.sort_values(by=["Year", "Athlete Name"], ascending=[False, True])

oo.sort_values(by=["Year", "Event", "Medal"], ascending=[True, True, False]).head(7)
