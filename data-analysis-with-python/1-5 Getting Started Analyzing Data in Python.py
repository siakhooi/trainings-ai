import pandas as pd

path='https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

df = pd.read_csv(path, header=None)
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers

print("df.dtypes")
print(df.dtypes)
print("df.describe()")
print(df.describe())
print("df.describe(include = all)")
print(df.describe(include = "all"))
print("df [[ 'length', 'compression-ratio' ]].describe()")
print(df [[ 'length', 'compression-ratio' ]].describe())
print("df.info")

df.info()

