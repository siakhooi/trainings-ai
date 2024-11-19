# import libraries
import csv
import torch
import pandas as pd

from IPython.display import display

# set data path
#Data_path = "../Documents/Tutors.csv"
Data_path = "./Tutors.csv"


# read the top 50 rows of the csv file into a pandas dataframe
# install Pandas library: pip3 install pandas
df = pd.read_csv(Data_path,nrows=50)


# display the pandas dataframe
display(df)


import torch.utils.data as data_utils
pd.to_numeric(df["sessions"])
sessions_df = pd.DataFrame(df['sessions'])

# creating tensor from session_df
train_tensor = torch.tensor(sessions_df['sessions'].values)

# printing out result
print(train_tensor)



# train_loader = data_utils.DataLoader(train_tensor, batch_size=10, shuffle=True)
