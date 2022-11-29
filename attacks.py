# FGSM
# PGD
import tensorflow as tf

from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.backend import sign


def fgsm_attack(X_test, y_test, model, epsilon):
    inp = tf.convert_to_tensor(X_test, dtype = tf.float32)
    imgv = tf.Variable(inp)
    with tf.GradientTape() as tape:
        tape.watch(imgv)
        predictions = model(imgv)
        loss = tf.keras.losses.MeanSquaredError()(y_test, predictions)
        grads = tape.gradient(loss,imgv)
    signed_grads = tf.sign(grads)
    inp = inp + (epsilon*signed_grads)

    return inp

def pgd_attack(X_test, y_test, model, iterations, alpha, epsilon):
    
    gen_img = tf.convert_to_tensor(X_test, dtype=tf.float32)
    gen_img = tf.clip_by_value(gen_img, X_test-epsilon, X_test+epsilon)
    x_temp = X_test

    for iters in range(iterations):
        imgv = tf.Variable(gen_img)
        with tf.GradientTape() as tape:
            tape.watch(imgv)
            predictions = model(imgv)
            loss = tf.keras.losses.MeanSquaredError()(y_test, predictions)
            grads = tape.gradient(loss,imgv)

        signed_grads = tf.sign(grads)
        gen_img = gen_img + (alpha*signed_grads)
        gen_img = tf.clip_by_value(gen_img, X_test-epsilon, X_test+epsilon)

        
    return gen_img