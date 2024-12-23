# 4.2 Prepare Input Data for Deep Learning

import pandas as pd
import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Data and review content
iris_data = pd.read_csv("iris.csv")

print("\nLoaded Data :\n------------------------------------")
print(iris_data.head())

# Use a Label encoder to convert String to numeric values
# for the target variable

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
iris_data["Species"] = label_encoder.fit_transform(iris_data["Species"])

# Convert input to numpy array
np_iris = iris_data.to_numpy()

# Separate feature and target variables
X_data = np_iris[:, 0:4]
Y_data = np_iris[:, 4]

print("\nFeatures before scaling :\n------------------------------------")
print(X_data[:5, :])
print("\nTarget before scaling :\n------------------------------------")
print(Y_data[:5])

# Create a scaler model that is fit on the input data.
scaler = StandardScaler().fit(X_data)

# Scale the numeric feature variables
X_data = scaler.transform(X_data)

# Convert target variable as a one-hot-encoding array
Y_data = tf.keras.utils.to_categorical(Y_data, 3)

print("\nFeatures after scaling :\n------------------------------------")
print(X_data[:5, :])
print("\nTarget after one-hot-encoding :\n------------------------------------")
print(Y_data[:5, :])

# Split training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.10)

print("\nTrain Test Dimensions:\n------------------------------------")
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


# 4.3 Creating a Model


from tensorflow import keras

# Number of classes in the target variable
NB_CLASSES = 3

# Create a sequencial model in Keras
model = tf.keras.models.Sequential()

# Add the first hidden layer
model.add(
    keras.layers.Dense(
        128,  # Number of nodes
        input_shape=(4,),  # Number of input variables
        name="Hidden-Layer-1",  # Logical name
        activation="relu",
    )
)  # activation function

# Add a second hidden layer
model.add(keras.layers.Dense(128, name="Hidden-Layer-2", activation="relu"))

# Add an output layer with softmax activation
model.add(keras.layers.Dense(NB_CLASSES, name="Output-Layer", activation="softmax"))

# Compile the model with loss & metrics
model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

# Print the model meta-data
model.summary()


# 4.4 Training and evaluating the model


# Make it verbose so we can see the progress
VERBOSE = 1

# Setup Hyper Parameters for training

# Set Batch size
BATCH_SIZE = 16
# Set number of epochs
EPOCHS = 10
# Set validation split. 20% of the training data will be used for validation
# after each epoch
VALIDATION_SPLIT = 0.2

print("\nTraining Progress:\n------------------------------------")

# Fit the model. This will perform the entire training cycle, including
# forward propagation, loss computation, backward propagation and gradient descent.
# Execute for the specified batch sizes and epoch
# Perform validation after each epoch
history = model.fit(
    X_train,
    Y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT,
)

print("\nAccuracy during Training :\n------------------------------------")
import matplotlib.pyplot as plt

# Plot accuracy of the model after each epoch.
pd.DataFrame(history.history)["accuracy"].plot(figsize=(8, 5))
plt.title("Accuracy improvements with Epoch")
plt.show()

# Evaluate the model against the test dataset and print results
print("\nEvaluation against Test Dataset :\n------------------------------------")
model.evaluate(X_test, Y_test)


# 4.5 Saving and Loading Models

# Saving a model

model.save("iris_save.keras")

# Loading a Model
loaded_model = keras.models.load_model("iris_save.keras")

# Print Model Summary
loaded_model.summary()


# 4.6 Predictions with Deep Learning Models


# Raw prediction data
prediction_input = [[6.6, 3.0, 4.4, 1.4]]

# Scale prediction data with the same scaling model
scaled_input = scaler.transform(prediction_input)

# Get raw prediction probabilities
raw_prediction = model.predict(scaled_input)
print("Raw Prediction Output (Probabilities) :", raw_prediction)

# Find prediction
prediction = np.argmax(raw_prediction)
print("Prediction is ", label_encoder.inverse_transform([prediction]))
