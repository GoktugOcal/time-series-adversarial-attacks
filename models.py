# LSTM
# BDLSTM
# Peephole LSTM
# BDSLTM encoder - LSTM decoder
# CNN
# CNN + LSTM

from tensorflow.keras.layers import Dense,Dropout,LSTM, Input, Flatten, Conv2D, MaxPool2D, Conv1D, MaxPooling1D, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RNN
from tensorflow.keras.experimental import PeepholeLSTMCell
# from keras.layers import CuDNNLSTM as LSTM

def single_layer_lstm(input_shape: tuple, output_len : int):

    model = Sequential()
    model.add(LSTM(32,activation="tanh", input_shape=input_shape))
    model.add(Dropout(0.15))
    model.add(Dense(output_len))
    model.summary()

    model.compile(optimizer="adam",loss="MSE")

    # model.name="Single Layer LSTM"

    return model

def double_layer_lstm(input_shape: tuple, output_len : int):

    model = Sequential()
    model.add(LSTM(32,activation="tanh", input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(32,activation="tanh", input_shape=input_shape))
    model.add(Dense(output_len))
    model.summary()

    model.compile(optimizer="adam",loss="MSE")

    # model.name="Double Layer LSTM"

    return model
    
def bidirectional_lstm(input_shape: tuple, output_len : int):

    model = Sequential()
    model.add(Bidirectional(LSTM(32,activation="tanh"), input_shape=input_shape))
    model.add(Dropout(0.15))
    model.add(Dense(output_len))
    model.summary()

    model.compile(optimizer="adam",loss="MSE")

    # model.name="Bidirectional LSTM"

    return model

def peephole_lstm(input_shape: tuple, output_len : int):

    model = Sequential()
    model.add(RNN([PeepholeLSTMCell(8)], input_shape=input_shape))
    model.add(Dropout(0.15))
    model.add(Dense(output_len))
    model.summary()

    model.compile(optimizer="adam",loss="MSE")

    # model.name="Bidirectional LSTM"

    return model

def cnn1d(input_shape: tuple, output_len : int):

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(output_len))
    model.compile(optimizer='adam', loss='mse')

    # model.name="CNN1D"
    
    return model

MODELS = {
    "Single Layer LSTM" : single_layer_lstm,
    "Double Layer LSTM" : double_layer_lstm,
    "Bidirectional LSTM" : bidirectional_lstm,
    # "Peephole LSTM" : peephole_lstm,
    "CNN1D" : cnn1d
}