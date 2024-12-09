from  Common_Experiment_Functions import *

## 3.4 Optimizers


accuracy_measures = {}

optimizer_list = ['sgd','rmsprop','adam','adagrad']
for optimizer in optimizer_list:

    model_config = base_model_config()
    X,Y = get_data()

    model_config["OPTIMIZER"] = optimizer
    model_name = "Optimizer-" + optimizer
    history=create_and_run_model(model_config,X,Y, model_name)

    accuracy_measures[model_name] = history.history["accuracy"]




plot_graph(accuracy_measures, "c3-4 Compare Optimizers")

