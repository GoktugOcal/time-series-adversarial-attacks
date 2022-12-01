import numpy as np
import math
from sklearn.metrics import mean_squared_error

def RMSE(true,pred):
    return math.sqrt(mean_squared_error(true,pred))

def MSE(true,pred):
    return mean_squared_error(true,pred)

def MAPE(true,pred):
    pred = pred[true != 0]
    true = true[true != 0]
    return np.mean((abs(true - pred) / true) * 100)