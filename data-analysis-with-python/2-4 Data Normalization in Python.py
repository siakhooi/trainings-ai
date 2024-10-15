import pandas as pd

path='https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

df = pd.read_csv(path, header=None)
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers

# Simple Feature Scaling

df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height']=df['height']/df['height'].max()
print(df[["length","width","height"]].head())

# Min Max
df['length'] = (df['length']-df['length'].min())/(df['length'].max()-df['length'].min())

# z-score
df['length'] = (df['length']-df['length'].mean())/df['length'].std()
