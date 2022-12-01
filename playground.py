import warnings
warnings.filterwarnings("ignore")

from data_loader import DataLoader
from models import MODELS
from misc.evaluation_metrics import *

dl = DataLoader()
info = dl.get_info()

print(info)

df = dl.load(0)
(X_train, y_train, X_test, y_test), scaler = dl.prepare_dataset(23*4,1)

print(dl.last_loaded_info["name"])
print(X_train.shape)
print(X_test.shape)
print()

model = MODELS[-1]((X_train.shape[1],X_train.shape[2]), 1)

model.summary()
# train LSTM model
model.compile(optimizer="adam",loss="MSE")
model.fit(X_train, y_train, epochs=1)

pred = model.predict(X_test)
pred = scaler.inverse_transform(pred)
y_test = scaler.inverse_transform(y_test)

y_test = y_test.reshape(1,-1)[0]
pred = pred.reshape(1,-1)[0]


print("RMSE :", RMSE(y_test,pred))
print("MSE :", MSE(y_test,pred))
print("MAPE :", MAPE(y_test,pred))
