from data_loader import DataLoader
from models import MODELS
from misc.evaluation_metrics import *

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience = 3)

import json
import warnings
warnings.filterwarnings("ignore")

dl = DataLoader()
info = dl.get_info()
df = dl.load(dataset_index = 0)

print(info)

model_info = {

}

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

        save_path = "models/" + row["name"] + "--" + model_name + ".h5"
        model_info[row["name"]][model_name] = {
            "RMSE" : round(RMSE(y_test_inv,pred),2),
            "MSE" : round(MSE(y_test_inv,pred),2),
            "MAPE" : round(MAPE(y_test_inv,pred),2),
            "path": save_path
        }

        model.save(save_path)
with open("models/model_info.json", "w") as outfile:
    json.dump(model_info, outfile, indent = 4)