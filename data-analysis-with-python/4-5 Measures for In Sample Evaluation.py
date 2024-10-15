import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

path='https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

df = pd.read_csv(path, header=None)
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers

lm = LinearRegression()

X = df[['highway-mpg']]
Y = df['price']
lm.fit(X,Y)

# get prediction
Yhat=lm.predict(X)

mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)
