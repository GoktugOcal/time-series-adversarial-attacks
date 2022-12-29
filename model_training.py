from misc.data_loader import DataLoader
from misc.models import MODELS
from misc.evaluation_metrics import *

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience = 3)

import json
import warnings
warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(
    prog = "Vanilla Model Training",
    description = "Trains the models defined in 'models.py' file and saves them to 'models/'"
)

parser.add_argument("--setting", required=True)
parser.add_argument("--datasets")
parser.add_argument("--models")
args = parser.parse_args()
settings = args.setting.lower()
datasets = args.datasets
models = args.models

settings = [item.strip() for item in settings.split(",")]

dl = DataLoader()
info = dl.get_info()
df = dl.load(dataset_index = 0)

if datasets:
    datasets = [item.strip() for item in datasets.split(",")]
    for dataset in datasets:
        if not dataset in info.name.tolist():
            raise ValueError("Dataset '" + dataset + "' does not exist in defined datasets.")

    info = info[info.name.isin(datasets)]

if models:
    models = [item.strip() for item in models.split(",")]
    for model in models:
        if not model in MODELS.keys():
            raise ValueError("Model '" + model + "' does not exist in defined models.")
    MODELS = dict(filter(lambda x: x[0] in models, MODELS.items()))

model_info = {

}

for setting in settings:

    if setting == "mto":

        for index, row in info.iterrows():
            model_info[row["name"]] = {}

            df = dl.load(index)

            if row["name"] == "Electricity Transformer Data - 15 min":
                (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23*4,1)

            elif row["name"] == "Metro Interstate Human Traffic Volume":
                (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23,1)

            elif row["name"] == "Beijing-Guanyuan Air-Quality":
                (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23,1)

            elif row["name"] == "Solar Generation - EnerjiSA":
                (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23,1)

            for (model_name, model) in MODELS.items():

                model = model((X_train.shape[1],X_train.shape[2]), 1)
                model.summary()
                # train LSTM model
                model.compile(optimizer="adam",loss="MSE")
                model.fit(
                    X_train,
                    y_train,
                    validation_split=0.25, 
                    epochs=50,
                    callbacks=[early_stopping])

                #Prediction Info
                pred = model.predict(X_test)
                pred = scaler.inverse_transform(pred).reshape(1,-1)[0]
                y_test_inv = scaler.inverse_transform(y_test).reshape(1,-1)[0]

                save_path = "models/many-to-one/" + row["name"] + "--" + model_name + ".h5"
                model_info[row["name"]][model_name] = {
                    "R2" : r2_score(y_test_inv,pred),
                    "MAE" : round(MAE(y_test_inv,pred),2),
                    "RMSE" : round(RMSE(y_test_inv,pred),2),
                    "MSE" : round(MSE(y_test_inv,pred),2),
                    "MAPE" : round(MAPE(y_test_inv,pred),2),
                    "SMAPE" : round(SMAPE(y_test_inv,pred),2),
                    "MDAPE" : round(MDAPE(y_test_inv,pred),2),
                    "path": save_path
                }

                model.save(save_path)
        with open("models/many-to-one/model_info.json", "w") as outfile:
            json.dump(model_info, outfile, indent = 4)

        print("Models have been saved to 'models/many-to-one/'")

    elif setting == "mtm":

        for index, row in info.iterrows():
            model_info[row["name"]] = {}

            df = dl.load(index)

            if row["name"] == "Electricity Transformer Data - 15 min":
                (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(24*7,12)

            elif row["name"] == "Metro Interstate Human Traffic Volume":
                (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(24*7,12)

            elif row["name"] == "Beijing-Guanyuan Air-Quality":
                (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(24*7,12)

            elif row["name"] == "Solar Generation - EnerjiSA":
                (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(24*7,12)

            for (model_name, model) in MODELS.items():

                model = model((X_train.shape[1],X_train.shape[2]), 12)
                model.summary()
                # train LSTM model
                model.compile(optimizer="adam",loss="MSE")
                model.fit(
                    X_train,
                    y_train,
                    validation_split=0.25, 
                    epochs=50,
                    callbacks=[early_stopping])

                #Prediction Info
                pred = model.predict(X_test)
                pred = scaler.inverse_transform(pred).reshape(1,-1)[0]
                y_test_inv = scaler.inverse_transform(y_test).reshape(1,-1)[0]

                save_path = "models/many-to-many/" + row["name"] + "--" + model_name + ".h5"
                model_info[row["name"]][model_name] = {
                    "R2" : r2_score(y_test_inv,pred),
                    "MAE" : round(MAE(y_test_inv,pred),2),
                    "RMSE" : round(RMSE(y_test_inv,pred),2),
                    "MSE" : round(MSE(y_test_inv,pred),2),
                    "MAPE" : round(MAPE(y_test_inv,pred),2),
                    "SMAPE" : round(SMAPE(y_test_inv,pred),2),
                    "MDAPE" : round(MDAPE(y_test_inv,pred),2),
                    "path": save_path
                }

                model.save(save_path)
        with open("models/many-to-many/model_info.json", "w") as outfile:
            json.dump(model_info, outfile, indent = 4)

        print("Models have been saved to 'models/many-to-one/'")