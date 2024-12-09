from  Common_Experiment_Functions import *

## 2.5 Choosing Activation Functions

accuracy_measures = {}

activation_list = ['relu','sigmoid','tanh', 'softmax']

for activation in activation_list:

    model_config = base_model_config()
    X,Y = get_data()

    model_config["HIDDEN_ACTIVATION"] = activation
    model_name = "Model-" + activation
    history=create_and_run_model(model_config,X,Y,model_name)

    accuracy_measures["Model-" + activation] = history.history["accuracy"]


plot_graph(accuracy_measures, "c2-5 Compare Activiation Functions")
