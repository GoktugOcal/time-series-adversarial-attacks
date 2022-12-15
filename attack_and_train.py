from data_loader import DataLoader
from models import MODELS
from misc.evaluation_metrics import *

from sklearn.utils import shuffle

# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience = 3)

from attacks import fgsm_attack, pgd_attack

from tqdm import tqdm
import json
import warnings
warnings.filterwarnings("ignore")

dl = DataLoader()
info = dl.get_info()
df = dl.load(dataset_index = 0)

print(info)

adv_model_info = {

}

with open("models/many-to-one/model_info.json", "r") as f:
    model_info = json.loads(f.read())

for index, row in info.iterrows():
    adv_model_info[row["name"]] = {}

    df = dl.load(index)

    if row["name"] == "Electricity Transformer Data - 15 min":
        (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23*4,1)

    elif row["name"] == "Metro Interstate Human Traffic Volume":
        (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23,1)

    elif row["name"] == "Beijing-Guanyuan Air-Quality":
        (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23,1)

    elif row["name"] == "Solar Generation - EnerjiSA":
        (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23,1)

    dataset = row["name"]

    for (model_name, model) in MODELS.items():
        adv_model_info[row["name"]][model_name] = {"FGSM" : {}, "PGD" : {}}
        model_path = model_info[row["name"]][model_name]["path"]
        org_model = load_model(model_path)

        # for attack in ["FGSM","PGD"]:
        for attack in ["PGD"]:

            if attack == "FGSM":
                # Attack to training set
                epsilon = 0.025
                X_adv = fgsm_attack(X_train, y_train, org_model, epsilon, np.inf, clip_min=0, clip_max=100000)
                # Attack to test set
                X_adv_test = fgsm_attack(X_test, y_test, org_model, epsilon, np.inf, clip_min=0, clip_max=100000)

                # Generate new training set
                adv_y_train = np.array(y_train.tolist()*2)
                adv_X_train = np.array(X_train.tolist() + X_adv.numpy().tolist())
                adv_X_train, adv_y_train = shuffle(adv_X_train,adv_y_train)

                # Get the model
                new_model = model((adv_X_train.shape[1],adv_X_train.shape[2]), 1)
                new_model.summary()
                # train LSTM model
                new_model.compile(optimizer="adam",loss="MSE")
                new_model.fit(
                    adv_X_train,
                    adv_y_train,
                    validation_split=0.25, 
                    epochs=1,
                    callbacks=[early_stopping])


                #Training accuracy
                pred = new_model.predict(adv_X_train)
                pred = scaler.inverse_transform(pred).reshape(1,-1)[0]
                y_train_inv = scaler.inverse_transform(adv_y_train).reshape(1,-1)[0]

                training_acc = {
                    "R2" : r2_score(y_train_inv,pred),
                    "MAE" : round(MAE(y_train_inv,pred),2),
                    "RMSE" : round(RMSE(y_train_inv,pred),2),
                    "MSE" : round(MSE(y_train_inv,pred),2),
                    "MAPE" : round(MAPE(y_train_inv,pred),2),
                    "SMAPE" : round(SMAPE(y_train_inv,pred),2),
                    "MDAPE" : round(MDAPE(y_train_inv,pred),2),
                }

                #Adversarial Test accuracy
                pred = new_model.predict(X_adv_test)
                pred = scaler.inverse_transform(pred).reshape(1,-1)[0]
                y_test_inv = scaler.inverse_transform(y_test).reshape(1,-1)[0]

                adv_test_acc = {
                    "R2" : r2_score(y_test_inv,pred),
                    "MAE" : round(MAE(y_test_inv,pred),2),
                    "RMSE" : round(RMSE(y_test_inv,pred),2),
                    "MSE" : round(MSE(y_test_inv,pred),2),
                    "MAPE" : round(MAPE(y_test_inv,pred),2),
                    "SMAPE" : round(SMAPE(y_test_inv,pred),2),
                    "MDAPE" : round(MDAPE(y_test_inv,pred),2),
                }

                save_path = "models/many-to-one/adv_trained_models/" + row["name"] + "-- trained_with_" + attack + "--" +model_name + ".h5"
                adv_model_info[row["name"]][model_name]["FGSM"] = {
                    "training accuracy": training_acc,
                    "adversarial testing accuracy" : adv_test_acc,
                    "path": save_path
                }
                new_model.save(save_path)


            elif attack == "PGD":
                # Attack to training set
                epsilon = 0.025
                alpha = 0.025
                iterations = 7
                X_adv = pgd_attack(X_train, y_train, org_model, iterations, alpha, epsilon, np.inf, clip_min=0, clip_max=100000)
                # Attack to test set
                X_adv_test = pgd_attack(X_test, y_test, org_model, iterations, alpha, epsilon, np.inf, clip_min=0, clip_max=100000)

                # Generate new training set
                adv_y_train = np.array(y_train.tolist()*2)
                adv_X_train = np.array(X_train.tolist() + X_adv.numpy().tolist())
                adv_X_train, adv_y_train = shuffle(adv_X_train,adv_y_train)

                # Get the model
                new_model = model((adv_X_train.shape[1],adv_X_train.shape[2]), 1)
                new_model.summary()
                # train LSTM model
                new_model.compile(optimizer="adam",loss="MSE")
                new_model.fit(
                    adv_X_train,
                    adv_y_train,
                    validation_split=0.25, 
                    epochs=1,
                    callbacks=[early_stopping])


                #Training accuracy
                pred = new_model.predict(adv_X_train)
                pred = scaler.inverse_transform(pred).reshape(1,-1)[0]
                y_train_inv = scaler.inverse_transform(adv_y_train).reshape(1,-1)[0]

                training_acc = {
                    "R2" : r2_score(y_train_inv,pred),
                    "MAE" : round(MAE(y_train_inv,pred),2),
                    "RMSE" : round(RMSE(y_train_inv,pred),2),
                    "MSE" : round(MSE(y_train_inv,pred),2),
                    "MAPE" : round(MAPE(y_train_inv,pred),2),
                    "SMAPE" : round(SMAPE(y_train_inv,pred),2),
                    "MDAPE" : round(MDAPE(y_train_inv,pred),2),
                }

                #Adversarial Test accuracy
                pred = new_model.predict(X_adv_test)
                pred = scaler.inverse_transform(pred).reshape(1,-1)[0]
                y_test_inv = scaler.inverse_transform(y_test).reshape(1,-1)[0]

                adv_test_acc = {
                    "R2" : r2_score(y_test_inv,pred),
                    "MAE" : round(MAE(y_test_inv,pred),2),
                    "RMSE" : round(RMSE(y_test_inv,pred),2),
                    "MSE" : round(MSE(y_test_inv,pred),2),
                    "MAPE" : round(MAPE(y_test_inv,pred),2),
                    "SMAPE" : round(SMAPE(y_test_inv,pred),2),
                    "MDAPE" : round(MDAPE(y_test_inv,pred),2),
                }

                save_path = "models/many-to-one/adv_trained_models/" + row["name"] + "-- trained_with_" + attack + "--" + model_name + ".h5"
                adv_model_info[row["name"]][model_name]["PGD"] = {
                    "training accuracy": training_acc,
                    "adversarial testing accuracy" : adv_test_acc,
                    "path": save_path
                }
                new_model.save(save_path)


with open("adv_examples/many-to-one/adv_training.json", "w") as outfile:
    json.dump(adv_samples, outfile)