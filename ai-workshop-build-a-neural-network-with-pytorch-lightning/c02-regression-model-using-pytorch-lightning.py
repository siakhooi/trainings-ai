import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl

from torch.utils.data import TensorDataset, DataLoader

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler, OneHotEncoder


class InsuranceDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int = 8):
        super().__init__()

        self.batch_size = batch_size

    def prepare_data(self):
        self.insurance_data = pd.read_csv("datasets/insurance.csv")

    def setup(self, stage = None):

        X = self.insurance_data.drop(columns = ["charges"])
        y = self.insurance_data["charges"]

        if stage == "fit" or stage is None:
            X_train, X_val, y_train, y_val = \
                train_test_split(X, y, test_size = 0.2, random_state = 123)

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

            y_train = y_train.to_numpy()
            y_val = y_val.to_numpy()

            ## Standard scaling features
            stdscaler = StandardScaler()
            X_train = stdscaler.fit_transform(X_train)
            X_val = stdscaler.transform(X_val)

            ## Min max scaling targets
            min_max_scaler = MinMaxScaler()
            y_train = min_max_scaler.fit_transform(y_train.reshape(-1, 1))
            y_val = min_max_scaler.transform(y_val.reshape(-1, 1))

            ## Convert to tensors
            self.train_inputs = torch.from_numpy(X_train).float()
            self.train_targets = torch.from_numpy(y_train.reshape(-1, 1)).float()
            self.val_inputs = torch.from_numpy(X_val).float()
            self.val_targets = torch.from_numpy(y_val.reshape(-1, 1)).float()

    def train_dataloader(self):
        train_dataset = TensorDataset(
            self.train_inputs, self.train_targets
        )
        train_loader = DataLoader(
            dataset = train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 4
        )

        return train_loader

    def val_dataloader(self):
        validation_dataset = TensorDataset(
            self.val_inputs, self.val_targets
        )
        validation_loader = DataLoader(
            dataset = validation_dataset,
            batch_size = self.batch_size,
            num_workers = 4
        )

        return validation_loader




insurance_dm = InsuranceDataModule()

insurance_dm.prepare_data()
insurance_dm.setup()



train_dl = insurance_dm.train_dataloader()
print(next(iter(train_dl)))



val_dl = insurance_dm.val_dataloader()

print(next(iter(val_dl)))


class LitRegressionModule(pl.LightningModule):

    def __init__(self, num_features, learning_rate = 0.01):
        super().__init__()

        self.layer_1 = nn.Linear(num_features, 16)
        self.layer_2 = nn.Linear(16, 32)
        self.layer_3 = nn.Linear(32, 16)
        self.layer_out = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.save_hyperparameters()

    def forward(self, inputs):

        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_out(x)

        return (x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        output = self.forward(x)

        loss_fn = nn.MSELoss(reduction = "mean")
        loss = loss_fn(output, y)

        self.log("train_loss", loss, prog_bar = True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        output = self.forward(x)
        loss_fn = nn.MSELoss(reduction = "mean")
        loss = loss_fn(output, y)

        self.log(f"val_loss", loss, on_epoch = True, prog_bar = True)

    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        x, y = batch

        return self(x)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr = self.hparams.learning_rate)





model = LitRegressionModule(num_features = 8)

print(model)





from pytorch_lightning.loggers import CSVLogger

insurance_dm = InsuranceDataModule()

logger = CSVLogger("logs", name = "Pytorch_lightning_training")

trainer = pl.Trainer(max_epochs = 50, logger = logger)

trainer.fit(model, datamodule = insurance_dm)






predictions = trainer.predict(model = model, dataloaders = insurance_dm.val_dataloader())

print(predictions)


y_pred_stacked = torch.cat(predictions)

print(y_pred_stacked[:10])




labels = []

for data, label in insurance_dm.val_dataloader():
    labels.extend(label)



y_true = torch.stack(labels)

print(y_true[:10])




from torchmetrics.regression import R2Score
from torchmetrics.regression import MeanSquaredError

MSE = MeanSquaredError()
r2score = R2Score()

print("Mean Squared Error :", round(MSE(y_pred_stacked, y_true).item(), 3))
print("R^2 :",round(r2score(y_pred_stacked, y_true).item(), 3))



metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

print(metrics.head(10))

print(metrics.tail(10))




del metrics["step"]

# Set the epoch column as the index, for easier plotting
metrics.set_index("epoch", inplace = True)

# Display the first few rows of the metrics table, excluding any columns with all NaN values
print(metrics.dropna(axis = 1, how = "all").head(10))

# Create a line plot of the training metrics using Seaborn
sns.lineplot(data = metrics)
plt.savefig("c02-metrics.png")
plt.close()
