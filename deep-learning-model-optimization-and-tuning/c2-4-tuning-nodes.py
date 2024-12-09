from  Common_Experiment_Functions import *

## 2.4 Nodes in a Layer

accuracy_measures = {}

for node_count in range(8,40,8):

    #have 2 hidden layers in the networks
    layer_list =[]
    for layer_count in range(2):
        layer_list.append(node_count)

    model_config = base_model_config()
    X,Y = get_data()

    model_config["HIDDEN_NODES"] = layer_list
    model_name = "Nodes-" + str(node_count)
    history=create_and_run_model(model_config,X,Y, model_name)

    accuracy_measures[model_name] = history.history["accuracy"]


plot_graph(accuracy_measures, "c2-4 Compare Nodes in a Layer")
