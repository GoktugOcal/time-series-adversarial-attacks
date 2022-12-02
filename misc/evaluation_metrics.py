import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def MAE(true,pred) -> np.array:
    return np.mean(abs(true - pred))

def RMSE(true,pred) -> np.array:
    return math.sqrt(mean_squared_error(true,pred))

def MSE(true,pred) -> np.array:
    return mean_squared_error(true,pred)

def MAPE(true,pred) -> np.array:
    pred = pred[true != 0]
    true = true[true != 0]
    return np.mean((abs(true - pred) / true) * 100)

def SMAPE(true, pred) -> np.array:
    return np.mean((abs(true - pred) / ((abs(true) + abs(pred))/2))*100)

def MDAPE(true, pred) -> np.array:
    return np.median((abs(true - pred) / true)*100)