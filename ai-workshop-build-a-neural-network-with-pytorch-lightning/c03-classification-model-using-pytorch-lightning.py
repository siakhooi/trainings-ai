import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import TensorDataset, DataLoader

from torchmetrics.functional import accuracy
from torchmetrics import F1Score

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder




bank_customer_churn_data = pd.read_csv("datasets/Churn_Modelling.csv")

print(bank_customer_churn_data.head())



print(bank_customer_churn_data .columns)


print(bank_customer_churn_data.info())



bank_customer_churn_data.dropna(inplace = True)

bank_customer_churn_data = bank_customer_churn_data.drop_duplicates()

print(bank_customer_churn_data.shape)




print(bank_customer_churn_data["Exited"].value_counts())





sns.countplot(data = bank_customer_churn_data  , x = "Exited")

plt.savefig("c03-data.png")
plt.close()




class BankCustomerChurnDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int = 8):
        super().__init__()

        self.batch_size = batch_size

    def prepare_data(self):
        self.bank_customer_churn_data = pd.read_csv("datasets/Churn_Modelling.csv")

        self.bank_customer_churn_data = self.bank_customer_churn_data.dropna()
        self. bank_customer_churn_data = self.bank_customer_churn_data.drop_duplicates()

    def setup(self, stage = None):
        X = self.bank_customer_churn_data.drop(
            columns = ["Exited", "RowNumber", "CustomerId", "Surname"]
        )
        y = self.bank_customer_churn_data["Exited"]

        if stage == "fit" or stage is None:
            X_train, X_val, y_train, y_val = \
                train_test_split(X, y, test_size = 0.2, random_state = 123)

            ## One hot encoding categorical features
            categorical_features = ['Geography', 'Gender']

            categorical_transformer = OneHotEncoder(
                handle_unknown = 'ignore', drop = 'first', sparse_output = False
            )

            preprocessor = ColumnTransformer(
                transformers = [('cat_tr', categorical_transformer, categorical_features)],
                remainder='passthrough'
            )

            X_train = preprocessor.fit_transform(X_train)
            X_val = preprocessor.transform(X_val)

            y_train, y_val = np.array(y_train), np.array(y_val)

            ## Scaling Inputs
            scaler = StandardScaler()

            inputs_train_array = scaler.fit_transform(X_train)
            inputs_val_array = scaler.transform(X_val)

            ## Input arrays and labels are converted into tensors
            self.train_inputs = torch.from_numpy(inputs_train_array).float()
            self.train_targets = torch.from_numpy(y_train.reshape(-1, 1)).float()
            self.val_inputs = torch.from_numpy(inputs_val_array).float()
            self.val_targets = torch.from_numpy(y_val.reshape(-1,1 )).float()

    def train_dataloader(self):
        train_dataset = TensorDataset(self.train_inputs, self.train_targets)
        train_loader = DataLoader(
            dataset = train_dataset, batch_size = self.batch_size, num_workers = 4
        )

        return train_loader

    def val_dataloader(self):
        validation_dataset = TensorDataset(self.val_inputs, self.val_targets)
        validation_loader = DataLoader(
            dataset = validation_dataset, batch_size = self.batch_size, num_workers = 4
        )

        return validation_loader





bank_customer_churn_dm = BankCustomerChurnDataModule()

bank_customer_churn_dm.prepare_data()
bank_customer_churn_dm.setup()




train_dl = bank_customer_churn_dm.train_dataloader()

sample_train_data_batch = next(iter(train_dl))

sample_train_data_batch





num_features = sample_train_data_batch[0].shape[1]

num_features




val_dl = bank_customer_churn_dm.val_dataloader()

next(iter(val_dl))




class LitBinaryClassificationModule(pl.LightningModule):

    def __init__(self, num_features, learning_rate = 0.001):
        super().__init__()

        self.layer1 = nn.Linear(num_features, 16)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(8, 4)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(4, 1)
        self.save_hyperparameters()

    def forward(self, x):

        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.output(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, y)

        self.log("train_loss", loss, on_step = False, on_epoch = True, prog_bar = True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, y)

        preds = torch.round(torch.sigmoid(logits))

        acc = accuracy(preds, y, task = "binary")

        f1_score = F1Score(task = "binary")
        f1 = f1_score(preds, y)

        self.log(f"val_loss", loss, on_epoch = True, prog_bar = True)
        self.log(f"val_acc", acc, on_epoch = True, prog_bar = True)
        self.log(f"val_f1", f1, on_epoch = True, prog_bar = True)

    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        x, y = batch

        return self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.hparams.learning_rate)





lit_clf_nn_model = LitBinaryClassificationModule(num_features = num_features)

print(lit_clf_nn_model)




from pytorch_lightning.loggers import CSVLogger

bank_customer_churn_dm = BankCustomerChurnDataModule()

logger = CSVLogger("logs", name = "pytorch_lightning_classification")

trainer = pl.Trainer(max_epochs = 20, logger = logger)

trainer.fit(lit_clf_nn_model , datamodule = bank_customer_churn_dm)






metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

print(metrics)




del metrics["step"]

# Set the epoch column as the index, for easier plotting
metrics.set_index("epoch", inplace = True)

# Display the first few rows of the metrics table, excluding any columns with all NaN values
print(metrics.dropna(axis = 1, how = "all").head())

# Create a line plot of the training metrics using Seaborn
sns.lineplot(data = metrics)

plt.savefig("c03-training.png")
plt.close()




y_pred_stacked = torch.cat(
    trainer.predict(model = lit_clf_nn_model , dataloaders = bank_customer_churn_dm.val_dataloader())
)

print(y_pred_stacked[:10])




y_pred = torch.round(torch.sigmoid(y_pred_stacked))

print(y_pred[:10])




labels = []

for data, label in bank_customer_churn_dm.val_dataloader():
    labels.extend(label)




y_true = torch.stack(labels)

print(y_true[:10])



print(y_true.shape)




from torchmetrics.classification import BinaryAccuracy
from torchmetrics import Recall, Precision, F1Score

acc = BinaryAccuracy()
prec = Precision(task = 'binary')
recall = Recall(task = 'binary')
f1 = F1Score(task = 'binary')

print("Accuracy_score :", round(acc(y_pred, y_true).item(),3))
print("Precision_score :", round(prec(y_pred,y_true).item(),3))
print("Recall_score :" , round(recall(y_pred,y_true).item(),3))
print("F1_score :" , round(f1(y_pred,y_true).item(),3))

