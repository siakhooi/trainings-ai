from  Common_Experiment_Functions import *

## 4.3. Regularization

accuracy_measures = {}

regularizer_list = ['l1','l2','l1l2']
for regularizer in regularizer_list:

    model_config = base_model_config()
    X,Y = get_data()

    model_config["REGULARIZER"] = regularizer
    model_config["EPOCHS"]=25
    model_name = "Regularizer-" + regularizer
    history=create_and_run_model(model_config,X,Y, model_name)

    #Switch to validation accuracy
    accuracy_measures[model_name] = history.history["val_accuracy"]


plot_graph(accuracy_measures, "c4-3 Compare Regularizers")
