from  Common_Experiment_Functions import *

## 2.2 Epoch and Batch Sizes


#Initialize the measures
accuracy_measures = {}

for batch_size in range(16,128,16):

    #Load default configuration
    model_config = base_model_config()
    #Acquire and process input data
    X,Y = get_data()

    #set epoch to 20
    model_config["EPOCHS"]=20
    #Set batch size to experiment value
    model_config["BATCH_SIZE"] = batch_size
    model_name = "Batch-Size-" + str(batch_size)
    history=create_and_run_model(model_config,X,Y,model_name)

    accuracy_measures[model_name] = history.history["accuracy"]


plot_graph(accuracy_measures, "c2-2 Compare Batch Size and Epoch")
