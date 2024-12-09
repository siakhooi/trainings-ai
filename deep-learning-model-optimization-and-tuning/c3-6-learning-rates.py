from  Common_Experiment_Functions import *

## 3.6. Learning Rates


accuracy_measures = {}

learning_rate_list = [0.001, 0.005,0.01,0.1,0.5]
for learning_rate in learning_rate_list:

    model_config = base_model_config()
    X,Y = get_data()

    model_config["LEARNING_RATE"] = learning_rate
    model_name="Learning-Rate-" + str(learning_rate)
    history=create_and_run_model(model_config,X,Y, model_name)

    #accuracy
    accuracy_measures[model_name] = history.history["accuracy"]


plot_graph(accuracy_measures, "c3-6 Compare Learning Rates")
