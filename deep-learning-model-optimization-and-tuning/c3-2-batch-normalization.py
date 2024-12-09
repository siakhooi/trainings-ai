from  Common_Experiment_Functions import *

## 3.2. Batch Normalization

accuracy_measures = {}

normalization_list = ['none','batch']
for normalization in normalization_list:

    model_config = base_model_config()
    X,Y = get_data()

    model_config["NORMALIZATION"] = normalization
    model_name="Normalization-" + normalization
    history=create_and_run_model(model_config,X,Y,model_name)

    accuracy_measures[model_name] = history.history["accuracy"]



plot_graph(accuracy_measures, "c3-2 Compare Normalization Techniques")
