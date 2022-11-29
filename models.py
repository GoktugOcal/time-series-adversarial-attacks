# LSTM
# BDLSTM
# Peephole LSTM
# BDSLTM encoder - LSTM decoder
# CNN
# CNN + LSTM

from keras.layers import Bidirectional

def single_layer_lstm(input_shape: tuple):

    model = Sequential()
    model.add(LSTM(input_shape[0],activation="tanh", input_shape=input_shape))
    model.add(Dropout(0.15))
    model.add(Dense(1))
    model.summary()

    model.compile(optimizer="adam",loss="MSE")

    return model

def double_layer_lstm(input_shape: tuple):

    model = Sequential()
    model.add(LSTM(input_shape[0],activation="tanh", input_shape=(input_shape, return_sequences=True)))
    model.add(Dropout(0.15))
    model.add(LSTM(input_shape[0],activation="tanh", input_shape=input_shape))
    model.add(Dense(1))
    model.summary()

    model.compile(optimizer="adam",loss="MSE")

    return model
    
def bidirectional_lstm(input_shape: tuple):

    model = Sequential()
    model.add(Bidirectional(LSTM(input_shape[0],activation="tanh"), input_shape=input_shape))
    model.add(Dropout(0.15))
    model.add(Dense(1))
    model.summary()

    model.compile(optimizer="adam",loss="MSE")

    return model

def cnn1d(input_shape: tuple):

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    return model