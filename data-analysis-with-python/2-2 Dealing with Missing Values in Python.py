import pandas as pd
import numpy as np

path='https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

df = pd.read_csv(path, header=None)
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers

# drop na
df.dropna(axis=0) # drop row
df.dropna(axis=1) # drop col
df.dropna(subset=["price"], axis=0, inplace = True)
df=df.dropna(subset=["price"], axis=0)

#replace missing value
df.replace("?", np.nan, inplace = True)
df=df.replace('?',np.NaN)

# replace with mean
mean=df["normalized-losses"].mean()
df["normalized-losses"].replace(np.nan, mean)


missing_data = df.isnull()
