import pandas as pd
import numpy as np
import json

from sklearn.preprocessing import MinMaxScaler

def load_data(stock, seq_len, out_len):
    X_train = []
    y_train = []
    for i in range(0, len(stock)-seq_len-out_len):
        X_train.append(stock.values[i: i+seq_len, 0])
        y_train.append(stock.values[i+seq_len: i+seq_len+out_len, 0])

    #Train-test split
    train_size = int(len(X_train)*0.7) #Set split size
    #Train
    X_test = X_train[train_size:len(X_train)]             
    y_test = y_train[train_size:len(y_train)]
    #Test
    X_train = X_train[:train_size]           
    y_train = y_train[:train_size]

    # convert to numpy array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    #4 reshape data to input into RNN models
    X_train = np.reshape(X_train, (train_size, seq_len, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], seq_len, 1))
    
    return [X_train, y_train, X_test, y_test]

class DataLoader:

    def __init__(self):
        with open('data/dataset_info.json') as f:
            dataset_info = json.loads(f.read())
        self.dataset_info = dataset_info
        self.last_loaded = None
        self.last_loaded_info = None

    def get_info(self):
        return pd.DataFrame(self.dataset_info)

    def load(self, dataset_index : int):
        
        selected_info = self.dataset_info[dataset_index]
        df = pd.read_csv(selected_info["path"])
        self.last_loaded = df
        self.last_loaded_info = selected_info
        return df

    def prepare_dataset(self, input_len, output_len):

        if self.last_loaded_info["name"] == "Electricity Transformer Data - 15 min":
            df = self.last_loaded
            df = df[["date","OT"]].set_index("date")
            df.columns = ["value"]
        
        elif self.last_loaded_info["name"] == "Metro Interstate Human Traffic Volume":
            df = self.last_loaded
            df = df[["date_time","traffic_volume"]].set_index("date_time")
            df.columns = ["value"]

        elif self.last_loaded_info["name"] == "Beijing-Guanyuan Air-Quality":
            df = self.last_loaded
            df["date_time"] = pd.to_datetime(df.year.astype(str) + '/' + df.month.astype(str) + '/' + df.day.astype(str) + " " + df.hour.astype(str) + ":00:00")
            df = df.fillna(method="bfill")
            
            df = df[["date_time","PM2.5"]].set_index("date_time")
            df.columns = ["value"]
        
        elif self.last_loaded_info["name"] == "Solar Generation - EnerjiSA":
            df = self.last_loaded.dropna()
            df["DateTime"] = pd.to_datetime([dt_item[0:2] + " " + dt_item[2:5] + " " + dt_item[5:] for dt_item in df["DateTime"]])

            df = df[["DateTime","Generation"]].set_index("DateTime")
            df.columns = ["value"]

        scaler = MinMaxScaler()
        df[:][:] = scaler.fit_transform(df)
        return load_data(df, input_len, output_len), scaler
