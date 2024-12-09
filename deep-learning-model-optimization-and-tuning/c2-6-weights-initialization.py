from  Common_Experiment_Functions import *

## 2.6. Weights Initialization

accuracy_measures = {}

initializer_list = ['random_normal','zeros','ones',"random_uniform"]
for initializer in initializer_list:

    model_config = base_model_config()
    X,Y = get_data()

    model_config["WEIGHTS_INITIALIZER"] = initializer
    model_name = "Model-" + initializer
    history=create_and_run_model(model_config,X,Y, model_name)

    accuracy_measures[model_name] = history.history["accuracy"]




plot_graph(accuracy_measures, "c2-6 Compare Weights Initializers")
