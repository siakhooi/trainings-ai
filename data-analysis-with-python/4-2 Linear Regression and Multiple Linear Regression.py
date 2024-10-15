import pandas as pd
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

print(Yhat)
print(Yhat[0:5])
print(lm.intercept_)
print(lm.coef_)

lm1=LinearRegression()
lm1

X1=df[['engine-size']]
Y1=df[['price']]
lm1.fit(X1,Y1)
lm1

lm1.coef_
lm1.intercept_


# Multiple Linear Regression

Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm = LinearRegression()
lm.fit(Z, df['price'])
Yhat=lm.predict(X)

print(lm.intercept_)
print(lm.coef_)

lm2 = LinearRegression()
lm2.fit(df[['normalized-losses' , 'highway-mpg']],df['price'])

lm2.coef_