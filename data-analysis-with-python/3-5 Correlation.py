import pandas as pd
import numpy as np
from scipy import stats
import skillsnetwork as sns
import matplotlib.pyplot as plt

path='https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

df = pd.read_csv(path, header=None)
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers

# 5. Correlation and Causation
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)

sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)

