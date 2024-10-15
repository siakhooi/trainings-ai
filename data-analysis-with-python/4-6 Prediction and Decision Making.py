import pandas as pd
import numpy as np
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

lm.predict(np.array(30.0).reshape(-1,1))

#

new_input=np.arange(1, 101, 1).reshape(-1, 1)
yhat=lm.predict(new_input)
yhat[0:5]

plt.plot(new_input, yhat)
plt.show()
