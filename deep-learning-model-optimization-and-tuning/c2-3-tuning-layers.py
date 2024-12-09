from  Common_Experiment_Functions import *

## 2.3. Layers in a Network

accuracy_measures = {}
layer_list =[]
for layer_count in range(1,6):

    #32 nodes in each layer
    layer_list.append(32)

    model_config = base_model_config()
    X,Y = get_data()

    model_config["HIDDEN_NODES"] = layer_list
    model_name = "Layers-" + str(layer_count)
    history=create_and_run_model(model_config,X,Y,model_name)

    accuracy_measures[model_name] = history.history["accuracy"]



plot_graph(accuracy_measures, "c2-3 Compare Hidden Layers")

