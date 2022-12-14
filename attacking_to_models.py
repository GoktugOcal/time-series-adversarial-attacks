from misc.data_loader import DataLoader
from misc.models import MODELS
from misc.attacks import fgsm_attack, pgd_attack
from misc.evaluation_metrics import *

from tensorflow.keras.models import load_model

from tqdm import tqdm
import json
import warnings
warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(
    prog = "Vanilla Model Training",
    description = "Trains the models defined in 'models.py' file and saves them to 'models/'"
)

parser.add_argument("--setting", required=True)
parser.add_argument("--attack", required=True)
parser.add_argument("--models")
args = parser.parse_args()
settings = args.setting.lower()
attacks = args.attack.lower()

settings = [item.strip() for item in settings.split(",")]
attacks = [item.strip() for item in attacks.split(",")]


dl = DataLoader()
info = dl.get_info()



for setting in settings:
    if setting == "mto":

        with open("models/many-to-one/model_info.json", "r") as f:
            model_info = json.loads(f.read())

        info = info[info.name.isin(list(model_info.keys()))]
        adv_samples = {}

        for index, row in tqdm(info.iterrows(), desc="dataset", position=0) :
            adv_samples[row["name"]] = {}

            df = dl.load(index)

            if row["name"] == "Electricity Transformer Data - 15 min":
                (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23*4,1)

            elif row["name"] == "Metro Interstate Human Traffic Volume":
                (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23,1)

            elif row["name"] == "Beijing-Guanyuan Air-Quality":
                (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23,1)

            elif row["name"] == "Solar Generation - EnerjiSA":
                (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23,1)


            for model_name in tqdm(model_info[row["name"]].keys(), desc="models", position=1):
                adv_samples[row["name"]][model_name] = {"FGSM": {}, "PGD": {}}

                model_path = model_info[row["name"]][model_name]["path"]
                model = load_model(model_path)
                
                if "fgsm" in attacks:
                    # FGSM attack
                    for epsilon in tqdm([0.01, 0.025, 0.05, 0.1], desc="FGSM", position=2):
                        X_adv = fgsm_attack(X_test, y_test, model, epsilon, np.inf, clip_min=0, clip_max=100000)

                        pred = model.predict(X_adv)
                        pred = scaler.inverse_transform(pred).reshape(1,-1)[0]
                        y_test_inv = scaler.inverse_transform(y_test).reshape(1,-1)[0]

                        adv_samples[row["name"]][model_name]["FGSM"]["epsilon=" + str(epsilon)] = {
                            # "data" : X_adv.numpy().reshape(X_adv.shape[0],X_adv.shape[1]),
                            "metrics" : {
                                "R2" : round(r2_score(y_test_inv,pred),3),
                                "MAE" : round(MAE(y_test_inv,pred),2),
                                "RMSE" : round(RMSE(y_test_inv,pred),2),
                                "MSE" : round(MSE(y_test_inv,pred),2),
                                "MAPE" : round(MAPE(y_test_inv,pred),2),
                                "SMAPE" : round(SMAPE(y_test_inv,pred),2),
                                "MDAPE" : round(MDAPE(y_test_inv,pred),2)
                            }
                        }

                if "pgd" in attacks:
                    # PGD attack
                    for alpha in tqdm([0.01, 0.025], desc="PGD", position=2):
                        for epsilon in [0.01, 0.025]:
                            iterations = 7
                            X_adv = pgd_attack(X_test, y_test, model, iterations, alpha, epsilon, np.inf, clip_min=0, clip_max=100000)

                            pred = model.predict(X_adv)
                            pred = scaler.inverse_transform(pred).reshape(1,-1)[0]
                            y_test_inv = scaler.inverse_transform(y_test).reshape(1,-1)[0]

                            adv_samples[row["name"]][model_name]["PGD"]["alpha=" + str(alpha) + " | epsilon=" + str(epsilon)] = {
                                # "data" : str(X_adv),
                                "metrics" : {
                                    "R2" : round(r2_score(y_test_inv,pred),3),
                                    "MAE" : round(MAE(y_test_inv,pred),2),
                                    "RMSE" : round(RMSE(y_test_inv,pred),2),
                                    "MSE" : round(MSE(y_test_inv,pred),2),
                                    "MAPE" : round(MAPE(y_test_inv,pred),2),
                                    "SMAPE" : round(SMAPE(y_test_inv,pred),2),
                                    "MDAPE" : round(MDAPE(y_test_inv,pred),2)
                                }
                            }

        with open("adv_examples/many-to-one/adv_gen_l_inf.json", "w") as outfile:
            json.dump(adv_samples, outfile)
        
        print("Model performances against adversarial attacks have been saved to 'adv_examples/many-to-one/adv_gen_l_inf.json'")
    
    elif setting == "mtm":

        with open("models/many-to-many/model_info.json", "r") as f:
            model_info = json.loads(f.read())

        info = info[info.name.isin(list(model_info.keys()))]
        adv_samples = {}

        for index, row in tqdm(info.iterrows(), desc="dataset", position=0) :
            adv_samples[row["name"]] = {}

            df = dl.load(index)

            if row["name"] == "Electricity Transformer Data - 15 min":
                (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(24*7,12)

            elif row["name"] == "Metro Interstate Human Traffic Volume":
                (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(24*7,12)

            elif row["name"] == "Beijing-Guanyuan Air-Quality":
                (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(24*7,12)

            elif row["name"] == "Solar Generation - EnerjiSA":
                (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(24*7,12)


            for model_name in tqdm(model_info[row["name"]].keys(), desc="models", position=1):
                adv_samples[row["name"]][model_name] = {"FGSM": {}, "PGD": {}}

                model_path = model_info[row["name"]][model_name]["path"]
                model = load_model(model_path)
                
                if "fgsm" in attacks:
                    # FGSM attack
                    for epsilon in tqdm([0.01, 0.025, 0.05, 0.1], desc="FGSM", position=2):
                        X_adv = fgsm_attack(X_test, y_test, model, epsilon, np.inf, clip_min=0, clip_max=100000)

                        pred = model.predict(X_adv)
                        pred = scaler.inverse_transform(pred).reshape(1,-1)[0]
                        y_test_inv = scaler.inverse_transform(y_test).reshape(1,-1)[0]

                        adv_samples[row["name"]][model_name]["FGSM"]["epsilon=" + str(epsilon)] = {
                            # "data" : X_adv.numpy().reshape(X_adv.shape[0],X_adv.shape[1]),
                            "metrics" : {
                                "R2" : round(r2_score(y_test_inv,pred),3),
                                "MAE" : round(MAE(y_test_inv,pred),2),
                                "RMSE" : round(RMSE(y_test_inv,pred),2),
                                "MSE" : round(MSE(y_test_inv,pred),2),
                                "MAPE" : round(MAPE(y_test_inv,pred),2),
                                "SMAPE" : round(SMAPE(y_test_inv,pred),2),
                                "MDAPE" : round(MDAPE(y_test_inv,pred),2)
                            }
                        }


                if "pgd" in attacks:

                    # PGD attack
                    for alpha in tqdm([0.01, 0.025], desc="PGD", position=2):
                        for epsilon in [0.01, 0.025]:
                            iterations = 7
                            X_adv = pgd_attack(X_test, y_test, model, iterations, alpha, epsilon, np.inf, clip_min=0, clip_max=100000)

                            pred = model.predict(X_adv)
                            pred = scaler.inverse_transform(pred).reshape(1,-1)[0]
                            y_test_inv = scaler.inverse_transform(y_test).reshape(1,-1)[0]

                            adv_samples[row["name"]][model_name]["PGD"]["alpha=" + str(alpha) + " | epsilon=" + str(epsilon)] = {
                                # "data" : str(X_adv),
                                "metrics" : {
                                    "R2" : round(r2_score(y_test_inv,pred),3),
                                    "MAE" : round(MAE(y_test_inv,pred),2),
                                    "RMSE" : round(RMSE(y_test_inv,pred),2),
                                    "MSE" : round(MSE(y_test_inv,pred),2),
                                    "MAPE" : round(MAPE(y_test_inv,pred),2),
                                    "SMAPE" : round(SMAPE(y_test_inv,pred),2),
                                    "MDAPE" : round(MDAPE(y_test_inv,pred),2)
                                }
                            }

        with open("adv_examples/many-to-many/adv_gen_l_inf.json", "w") as outfile:
            json.dump(adv_samples, outfile)

        print("Model performances against adversarial attacks have been saved to 'adv_examples/many-to-many/adv_gen_l_inf.json'")
        