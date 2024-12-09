from  Common_Experiment_Functions import *

## 5.2 Acquire and Process Data

import pandas as pd
import os
import tensorflow as tf

def get_rca_data():
    #Load the data file into a Pandas Dataframe
    symptom_data = pd.read_csv("root_cause_analysis.csv")

    #Explore the data loaded
    #print(symptom_data.dtypes)
    #symptom_data.head()

    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split

    label_encoder = preprocessing.LabelEncoder()
    symptom_data['ROOT_CAUSE'] = label_encoder.fit_transform(
                                    symptom_data['ROOT_CAUSE'])

    #Convert Pandas DataFrame to a numpy vector
    np_symptom = symptom_data.to_numpy().astype(float)

    #Extract the feature variables (X)
    X_data = np_symptom[:,1:8]

    #Extract the target variable (Y), conver to one-hot-encoding
    Y_data=np_symptom[:,8]
    Y_data = tf.keras.utils.to_categorical(Y_data,3)

    return X_data,Y_data

## 5.3. Tuning the network

### 5.3.1. Layers in the network

accuracy_measures = {}
layer_list =[]
for layer_count in range(1,6):

    #32 nodes in each layer
    layer_list.append(32)

    model_config = base_model_config()
    X,Y = get_rca_data()

    model_config["HIDDEN_NODES"] = layer_list
    model_name = "Layers-" + str(layer_count)
    history=create_and_run_model(model_config,X,Y,model_name)

    accuracy_measures[model_name] = history.history["accuracy"]

plot_graph(accuracy_measures, "c5-3-1 Compare Hidden Layers")


### 5.3.2 Nodes in a Layer

accuracy_measures = {}

node_increment=8

for node_count in range(1,5):

    #have 2 hidden layers in the networks as selected above
    layer_list =[]
    for layer_count in range(2):
        layer_list.append(node_count * node_increment)

    model_config = base_model_config()
    X,Y = get_rca_data()

    model_config["HIDDEN_NODES"] = layer_list
    model_name = "Nodes-" + str(node_count * node_increment)
    history=create_and_run_model(model_config,X,Y, model_name)

    accuracy_measures[model_name] = history.history["accuracy"]

plot_graph(accuracy_measures, "c5-3-2 Compare Nodes in a Layer")

## 5.4. Tuning Back Propagation

## 5.4.1. Optimizers

accuracy_measures = {}

optimizer_list = ['sgd','rmsprop','adam','adagrad']
for optimizer in optimizer_list:

    model_config = base_model_config()
    X,Y = get_rca_data()

    model_config["OPTIMIZER"] = optimizer
    model_name = "Optimizer-" + optimizer
    history=create_and_run_model(model_config,X,Y, model_name)

    accuracy_measures[model_name] = history.history["accuracy"]

plot_graph(accuracy_measures, "c5-4-1 Compare Optimizers")

### 5.4.2. Learning Rates


accuracy_measures = {}

learning_rate_list = [0.001, 0.005,0.01,0.1,0.5]
for learning_rate in learning_rate_list:

    model_config = base_model_config()
    X,Y = get_rca_data()

    #Fix Optimizer to the one chosen above
    model_config["OPTIMIZER"]="rmsprop"
    model_config["LEARNING_RATE"] = learning_rate
    model_name="Learning-Rate-" + str(learning_rate)
    history=create_and_run_model(model_config,X,Y, model_name)

    #Using validation accuracy
    accuracy_measures[model_name] = history.history["accuracy"]


plot_graph(accuracy_measures, "c5-4-2 Compare Learning Rates")

## 5.5. Avoiding Overfitting

### 5.5.1. Regularizer


accuracy_measures = {}

regularizer_list = [None,'l1','l2','l1l2']
for regularizer in regularizer_list:

    model_config = base_model_config()
    X,Y = get_rca_data()

    model_config["REGULARIZER"] = regularizer
    model_name = "Regularizer-" + str(regularizer)
    history=create_and_run_model(model_config,X,Y, model_name)

    #Switch to validation accuracy
    accuracy_measures[model_name] = history.history["val_accuracy"]


plot_graph(accuracy_measures, "c5-5-1 Compare Regularizers")

### 5.5.2. Dropout


accuracy_measures = {}

dropout_list = [0.0, 0.1, 0.2, 0.5]
for dropout in dropout_list:

    model_config = base_model_config()
    X,Y = get_rca_data()

    #Use the regularizer chosen above
    model_config["REGULARIZER"] = "l2"
    model_config["DROPOUT_RATE"] = dropout
    model_name="Dropout-" + str(dropout)
    history=create_and_run_model(model_config,X,Y, model_name)

    #Using validation accuracy
    accuracy_measures[model_name] = history.history["val_accuracy"]


plot_graph(accuracy_measures, "c5-5-2 Compare Dropout")


## 5.6. Building the final model



accuracy_measures = {}

#Base Minimal Model
model_config = base_model_config()
model_config["HIDDEN_NODES"] = [16]
model_config["NORMALIZATION"] = None
model_config["OPTIMIZER"] = "rmsprop"
model_config["LEARNING_RATE"] = 0.001
model_config["REGULARIZER"]=None
model_config["DROPOUT_RATE"] = 0.0

X,Y = get_rca_data()

model_name = "Base-Model-" + str(layer_count)

history=create_and_run_model(model_config,X,Y,model_name)

accuracy_measures[model_name] = history.history["accuracy"]

#Adding all optimizations
model_config = base_model_config()
model_config["HIDDEN_NODES"] = [32,32]
model_config["NORMALIZATION"] = "batch"
model_config["OPTIMIZER"] = "rmsprop"
model_config["LEARNING_RATE"] = 0.001
model_config["REGULARIZER"]="l2"
model_config["DROPOUT_RATE"] = 0.2

X,Y = get_rca_data()

model_name = "Optimized-Model-" + str(layer_count)

history=create_and_run_model(model_config,X,Y,model_name)

accuracy_measures[model_name] = history.history["accuracy"]


plot_graph(accuracy_measures, "c5-6 Compare Cumulative Improvements")

