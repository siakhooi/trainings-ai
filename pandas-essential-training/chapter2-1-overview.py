import pandas as pd

filename = "olympics_1896_2004.csv"

oo = pd.read_csv(filename, skiprows=5)

print(oo.shape)

print(oo.head())
print(oo.head(3))
print(oo.tail())
print(oo.tail(3))
print(oo.sample(5))
print(oo.info())
print(oo.describe())

print(pd.get_option("display.max_rows"))
pd.set_option("display.max_rows", None)
pd.set_option("display.max_rows", 60)

print(oo)

c=pd.get_option("display.max_columns")
print(f'display.max_columns: {c}')
print(oo)
pd.set_option("display.max_columns", 2)
print(oo)
pd.set_option("display.max_columns", 0)

print(pd.get_option("display.width"))
pd.set_option("display.width", 100)
print(oo)
