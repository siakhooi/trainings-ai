import torch

print(torch.__version__)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler , OneHotEncoder


insurance_data = pd.read_csv("datasets/insurance.csv")

insurance_data.head()

print(insurance_data.shape)

insurance_data.info()

insurance_data.columns

sns.histplot(insurance_data["charges"])

plt.savefig("c01-insurance-charges-histogram.png")
plt.close()

sns.boxplot(y = insurance_data["charges"], x = insurance_data["smoker"])
plt.savefig("c01-insurance-charges-smoker-boxplot.png")
plt.close()

sns.scatterplot(y = insurance_data["charges"], x = insurance_data["age"])
plt.savefig("c01-insurance-charges-age-scatterplot.png")
plt.close()


X = insurance_data.drop(columns = ["charges"])
y = insurance_data["charges"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 123)

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

categorical_features = ["sex", "smoker", "region"]

categorical_transformer = OneHotEncoder(
    handle_unknown = "ignore", drop = "first", sparse_output = False
)

preprocessor = ColumnTransformer(
    transformers = [("cat_tr", categorical_transformer, categorical_features)],
    remainder = "passthrough"
)


X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)

print(X_train.shape, X_val.shape)


print(X_train)

pd.DataFrame(X_train, columns = preprocessor.get_feature_names_out()).T


y_train = y_train.to_numpy()
y_val = y_val.to_numpy()

y_train[:10]


stdscaler = StandardScaler()

X_train = stdscaler.fit_transform(X_train)
X_val = stdscaler.transform(X_val)

print(X_train)


y_train.reshape(-1, 1)



min_max_scaler = MinMaxScaler()

y_train = min_max_scaler.fit_transform(y_train.reshape(-1, 1))

y_val = min_max_scaler.transform(y_val.reshape(-1, 1))


train_inputs = torch.from_numpy(X_train).float()
train_targets = torch.from_numpy(y_train.reshape(-1, 1)).float()

train_inputs.shape, train_targets.shape



class SimpleNeuralNet(nn.Module):

    # Initialize the layers
    def __init__(self, num_features):

        super().__init__()

        self.linear1 = nn.Linear(num_features, 1)

    # Perform the computation
    def forward(self, x):

        x = self.linear1(x)

        return x





## Neural Network with more layers
class NeuralNetRegression(nn.Module):
    # Initialize the layers
    def __init__(self, num_features):
        super(NeuralNetRegression, self).__init__()

        self.layer_1 = nn.Linear(num_features, 16)
        self.layer_2 = nn.Linear(16, 32)
        self.layer_3 = nn.Linear(32, 16)
        self.layer_out = nn.Linear(16, 1)

        self.relu = nn.ReLU()
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_out(x)
        return (x)
    def predict(self, test_inputs):
        x = self.relu(self.layer_1(test_inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_out(x)
        return (x)

model = SimpleNeuralNet(num_features=8)

print(model)



for layer in model.children():

    if isinstance(layer, nn.Linear):
        print(layer.state_dict()["weight"])
        print(layer.state_dict()["bias"])



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

count_parameters(model)


import torch.nn.functional as F

loss_fn = F.mse_loss


loss = loss_fn(model(train_inputs), train_targets)

print(loss)




preds = model(train_inputs)

print(preds)



from torchmetrics.regression import R2Score
from torchmetrics.regression import MeanSquaredError

MSE = MeanSquaredError()

r2score = R2Score()

print("Mean Squared Error :", MSE(preds, train_targets).item())
print("R^2 :", r2score(preds, train_targets).item())



from torch.utils.data import TensorDataset, DataLoader

train_ds = TensorDataset(train_inputs, train_targets)
train_ds[:5]



batch_size = 8
train_dl = DataLoader(train_ds, batch_size, shuffle = True)
next(iter(train_dl))




val_inputs = torch.from_numpy(X_val).float()
val_targets = torch.from_numpy(y_val.reshape(-1, 1)).float()



val_ds = TensorDataset(val_inputs, val_targets)

val_ds[:5]



val_dl = DataLoader(val_ds, batch_size)

next(iter(val_dl))



loss_stats = {
    "train": [],
    "val": []
}

num_epochs = 100



device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")



model = SimpleNeuralNet(num_features=8).to(device)

print(model)



optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)




for epoch in range(num_epochs):
    # TRAINING
    train_epoch_loss = 0

    model.train()

    for X_train_batch, y_train_batch in train_dl:

        optimizer.zero_grad()

        # Move data to the same device as the model
        X_train_batch, y_train_batch = \
            X_train_batch.to(device), y_train_batch.to(device)

        # Generate predictions and compute loss
        preds = model(X_train_batch)

        train_loss = loss_fn(preds, y_train_batch)

        # Perform gradient descent
        train_loss.backward()

        optimizer.step()

        train_epoch_loss += train_loss.item()

    # VALIDATION
    with torch.no_grad():

        val_epoch_loss = 0

        model.eval()

        for X_val_batch, y_val_batch in val_dl:

            X_val_batch, y_val_batch = \
                X_val_batch.to(device), y_val_batch.to(device)

            y_val_pred = model(X_val_batch)

            val_loss = loss_fn(y_val_pred, y_val_batch)

            val_epoch_loss += val_loss.item()

    train_loss = train_epoch_loss / len(train_dl)
    val_loss = val_epoch_loss / len(val_dl)

    loss_stats["train"].append(train_loss)
    loss_stats["val"].append(val_loss)

    print(f'Epoch {epoch+0:01}: | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}')



train_val_loss_df = pd.DataFrame.from_dict(loss_stats). \
    reset_index().melt(id_vars = ["index"]). \
    rename(columns = {"index": "epochs"})

train_val_loss_df.head()




train_val_loss_df.tail()




plt.figure(figsize = (10, 5))

sns.lineplot(data = train_val_loss_df,
             x = "epochs", y = "value",
             hue = "variable"
).set_title("Train-Val Loss vs Epoch")
plt.savefig("c01-train-val-loss-vs-epoch.png")
plt.close()



y_true = []
y_pred = []

with torch.no_grad():

    model.eval()

    for X_batch, y_batch in val_dl:
        X_batch = X_batch.to(device)

        y_test_pred = model(X_batch)

        y_true.extend(y_batch)
        y_pred.extend(y_test_pred)



y_true[:10]



y_true_stacked = torch.stack((y_true))

y_true_stacked[:10]



y_pred_stacked = torch.stack((y_pred))

y_pred_stacked[:10]



MSE = MeanSquaredError().to(device)
r2score = R2Score().to(device)

print("Mean Squared Error :", round(MSE(y_pred_stacked.to(device), y_true_stacked.to(device)).item(), 3))

print("R^2 :", round(r2score(y_pred_stacked.to(device), y_true_stacked.to(device)).item(), 3))


