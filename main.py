import warnings
warnings.filterwarnings("ignore")

from data_loader import DataLoader
from models import MODELS

dl = DataLoader()
info = dl.get_info()
df = dl.load(dataset_index = 0)

print(info)

for index, row in info.iterrows():

    df = dl.load(index)

    if row["name"] == "Electricity Transformer Data - 15 min":
        (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23*4,1)

    elif row["name"] == "Metro Interstate Human Traffic Volume":
        (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23,1)

    elif row["name"] == "Beijing-Guanyuan Air-Quality":
        (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23,1)

    elif row["name"] == "Solar Generation - EnerjiSA":
        (X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23,1)


    print(dl.last_loaded_info["name"])
    print(X_train.shape)
    print(X_test.shape)
    print()