import pandas as pd

filename = "olympics_1896_2004.csv"

oo = pd.read_csv(filename, skiprows=5)
oo.Gender = oo.Gender.astype("category")
oo['Event Gender'] = oo['Event Gender'].astype("category")
medal_order = ["Bronze", "Silver", "Gold"]
oo.Medal = pd.Categorical(oo.Medal, categories=medal_order, ordered=True)


df = pd.read_csv(filename, skiprows=5)

print(f"Medal series memory usage using dtype category: {oo.Medal.memory_usage(deep=True)}")
print(f"Medal series memory usage using dtype object: {df.Medal.memory_usage(deep=True)}")

print(oo.Medal.memory_usage(deep=True) / df.Medal.memory_usage(deep=True))


print(f"Gender series memory usage using dtype category: {oo.Gender.memory_usage(deep=True)}")
print(f"Gender series memory usage using dtype object: {df.Gender.memory_usage(deep=True)}")
print(f"Gender series memory usage using dtype category: {oo['Event Gender'].memory_usage(deep=True)}")
print(f"Gender series memory usage using dtype object: {df['Event Gender'].memory_usage(deep=True)}")


oo.City = oo.City.astype("string")
oo.Sport = oo.Sport.astype("string")
oo.Discipline = oo.Discipline.astype("string")
oo["Athlete Name"] = oo["Athlete Name"].astype("string")
oo.NOC = oo.NOC.astype("string")
oo.Event = oo.Event.astype("string")
print(oo.dtypes)

print(f"City series memory usage using dtype string: {oo.City.memory_usage(deep=True)}")
print(f"City series memory usage using dtype object: {df.City.memory_usage(deep=True)}")
