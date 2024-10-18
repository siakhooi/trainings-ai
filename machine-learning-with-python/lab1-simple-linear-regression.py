import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

num = 1


def plt_save_and_close(label=""):
    global num
    plt.savefig(f"figure lab1-simple-linear-regression-{num}-{label}.png")
    num += 1
    plt.clf()
    plt.cla()
    plt.close()


df = pd.read_csv("resources/FuelConsumption.csv")
df.head()
df.describe()

cdf = df[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]
cdf.head(9)

viz = cdf[["CYLINDERS", "ENGINESIZE", "CO2EMISSIONS", "FUELCONSUMPTION_COMB"]]
viz.hist()
plt.show()
plt_save_and_close()

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()
plt_save_and_close()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
plt_save_and_close()


plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()
plt_save_and_close()

# # Creating train and test dataset

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# # Simple Regression Model

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="blue")
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
plt_save_and_close()


regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["ENGINESIZE"]])
train_y = np.asanyarray(train[["CO2EMISSIONS"]])
regr.fit(train_x, train_y)
# The coefficients
print("Coefficients: ", regr.coef_)
print("Intercept: ", regr.intercept_)

# Coefficients:  [[38.80325496]]
# Intercept:  [126.31252947]

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="blue")
plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], "-r")
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt_save_and_close()


test_x = np.asanyarray(test[["ENGINESIZE"]])
test_y = np.asanyarray(test[["CO2EMISSIONS"]])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y, test_y_))

# Mean absolute error: 22.39
# Residual sum of squares (MSE): 936.76
# R2-score: 0.78

train_x = train[["FUELCONSUMPTION_COMB"]]
test_x = test[["FUELCONSUMPTION_COMB"]]

regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)

predictions = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(predictions - test_y)))

# Mean absolute error: 19.83
