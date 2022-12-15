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


index = 0

dataset_name = info.name[index]
print(dataset_name)

df = dl.load(index)


if dataset_name == "Electricity Transformer Data - 15 min":
    (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23*4,1)

elif dataset_name == "Metro Interstate Human Traffic Volume":
    (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23,1)

elif dataset_name == "Beijing-Guanyuan Air-Quality":
    (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23,1)

elif dataset_name == "Solar Generation - EnerjiSA":
    (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23,1)

print(X_train.shape)