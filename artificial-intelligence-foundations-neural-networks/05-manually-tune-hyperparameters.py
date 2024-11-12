# Step 1: Data Preparation

# Import the necessary libraries

# For Data loading, Exploraotry Data Analysis, Graphing
import pandas as pd  # Pandas for data processing libraries
import numpy as np  # Numpy for mathematical functions

import matplotlib.pyplot as plt  # Matplotlib for visualization tasks
import seaborn as sns  # Seaborn for data visualization library based on matplotlib.

# %matplotlib inline

import sklearn  # ML tasks
from sklearn.model_selection import train_test_split  # Split the dataset
from sklearn.metrics import mean_squared_error  # Calculate Mean Squared Error

# Build the Network
from tensorflow import keras
from keras.models import Sequential

# from tensorflow.keras.models import Sequential
from keras.layers import Dense


# Next, you read the dataset into a Pandas dataframe.

url = "https://github.com/LinkedInLearning/artificial-intelligence-foundations-neural-networks-4381282/blob/main/Advertising_2023.csv?raw=true"
advertising_df = pd.read_csv(url, index_col=0)


# Pandas info() function is used to get a concise summary of the dataframe.
advertising_df.info()


### Get summary of statistics of the data
advertising_df.describe()


# shape of dataframe - 1199 rows, five columns
advertising_df.shape


# The isnull() method is used to check and manage NULL values in a data frame.
advertising_df.isnull().sum()


# Exploratory Data Analysis (EDA)

## Plot the heatmap so that the values are shown.

plt.figure(figsize=(10, 5))
sns.heatmap(advertising_df.corr(), annot=True, vmin=0, vmax=1, cmap="ocean")

plt.savefig("figure 05-01.png")


# create a correlation matrix
corr = advertising_df.corr()
plt.figure(figsize=(10, 5))
sns.heatmap(
    corr[(corr >= 0.5) | (corr <= -0.7)],
    cmap="viridis",
    vmax=1.0,
    vmin=-1.0,
    linewidths=0.1,
    annot=True,
    annot_kws={"size": 8},
    square=True,
)
plt.tight_layout()
# display(plt.show())
plt.savefig("figure 05-02.png")

advertising_df.corr()


### Visualize Correlation

# Generate a mask for the upper triangle
mask = np.zeros_like(advertising_df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    advertising_df.corr(),
    mask=mask,
    cmap=cmap,
    vmax=0.9,
    square=True,
    linewidths=0.5,
    ax=ax,
)


plt.savefig("figure 05-03.png")


"""=== Show the linear relationship between features  and sales Thus, it provides that how the scattered
      they are and which features has more impact in prediction of house price. ==="""

# visiualize all variables  with sales
from scipy import stats

# creates figure
plt.figure(figsize=(18, 18))

for i, col in enumerate(
    advertising_df.columns[0:13]
):  # iterates over all columns except for price column (last one)
    plt.subplot(5, 3, i + 1)  # each row three figure
    x = advertising_df[col]  # x-axis
    y = advertising_df["sales"]  # y-axis
    plt.plot(x, y, "o")

    # Create regression line
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color="red")
    plt.xlabel(col)  # x-label
    plt.ylabel("sales")  # y-label


plt.savefig("figure 05-04.png")

# Training a Linear Regression Model


X = advertising_df[["digital", "TV", "radio", "newspaper"]]
y = advertising_df["sales"]


"""=== Noramlization the features. Since it is seen that features have different ranges, it is best practice to
normalize/standarize the feature before using them in the model ==="""

# feature normalization
normalized_feature = keras.utils.normalize(X.values)


# Import train_test_split function from sklearn.model_selection
from sklearn.model_selection import train_test_split

# Split up the data into a training set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=101
)


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Step 2: Build Network


## Build Model (Building a three layer network - with one hidden layer)
model = Sequential()
model.add(
    Dense(4, input_dim=4, activation="relu")
)  # You don't have to specify input size.Just define the hidden layers
model.add(Dense(3, activation="relu"))
model.add(Dense(1))

# Compile Model
model.compile(optimizer="adam", loss="mse", metrics=["mse"])

#  Fit the Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=32)


# Visualization


## Plot a graph of model loss # show the graph of model loss in trainig and validation

plt.figure(figsize=(15, 8))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss (MSE) on Training and Validation Data")
plt.ylabel("Loss-Mean Squred Error")
plt.xlabel("Epoch")
plt.legend(["Val Loss", "Train Loss"], loc="upper right")
plt.show()

plt.savefig("figure 05-05.png")


# Step 3: Tune the Neural Network Hyperparameters

## Build Model
model = Sequential()
model.add(Dense(4, input_dim=4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(3, activation="relu"))
model.add(Dense(1))

# Compile Model
model.compile(optimizer="adam", loss="mse", metrics=["mse"])

#  Fit the Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)


plt.figure(figsize=(15, 8))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss (MSE) on Training and Validation Data")
plt.ylabel("Loss-Mean Squred Error")
plt.xlabel("Epoch")
plt.legend(["Val Loss", "Train Loss"], loc="upper right")
plt.show()


plt.savefig("figure 05-06.png")
