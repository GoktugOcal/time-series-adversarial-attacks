# FGSM
# PGD
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.backend import sign

def clip_eta(eta, norm, eps):
    """
    Helper function to clip the perturbation to epsilon norm ball.
    :param eta: A tensor with the current perturbation.
    :param norm: Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param eps: Epsilon, bound of the perturbation.
    """

    # Clipping perturbation eta to self.norm norm ball
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")
    axis = list(range(1, len(eta.get_shape())))
    avoid_zero_div = 1e-12
    if norm == np.inf:
        eta = tf.clip_by_value(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("")
            # This is not the correct way to project on the L1 norm ball:
            # norm = tf.maximum(avoid_zero_div, reduce_sum(tf.abs(eta), reduc_ind, keepdims=True))
        elif norm == 2:
            # avoid_zero_div must go inside sqrt to avoid a divide by zero in the gradient through this operation
            norm = tf.sqrt(
                tf.maximum(
                    avoid_zero_div, tf.reduce_sum(tf.square(eta), axis, keepdims=True)
                )
            )
        # We must *clip* to within the norm ball, not *normalize* onto the surface of the ball
        factor = tf.minimum(1.0, tf.math.divide(eps, norm))
        eta = eta * factor
    return eta

def optimize_linear(grad, eps, norm=np.inf):
    """
    Solves for the optimal input to a linear function under a norm constraint.
    Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)
    :param grad: tf tensor containing a batch of gradients
    :param eps: float scalar specifying size of constraint region
    :param norm: int specifying order of norm
    :returns:
      tf tensor containing optimal perturbation
    """

    # Convert the iterator returned by `range` into a list.
    axis = list(range(1, len(grad.get_shape())))
    avoid_zero_div = 1e-12
    if norm == np.inf:
        # Take sign of gradient
        optimal_perturbation = tf.sign(grad)
        # The following line should not change the numerical results. It applies only because
        # `optimal_perturbation` is the output of a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the perturbation has a non-zero derivative.
        optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    elif norm == 1:
        abs_grad = tf.abs(grad)
        sign = tf.sign(grad)
        max_abs_grad = tf.reduce_max(abs_grad, axis, keepdims=True)
        tied_for_max = tf.dtypes.cast(
            tf.equal(abs_grad, max_abs_grad), dtype=tf.float32
        )
        num_ties = tf.reduce_sum(tied_for_max, axis, keepdims=True)
        optimal_perturbation = sign * tied_for_max / num_ties
    elif norm == 2:
        square = tf.maximum(
            avoid_zero_div, tf.reduce_sum(tf.square(grad), axis, keepdims=True)
        )
        optimal_perturbation = grad / tf.sqrt(square)
    else:
        raise NotImplementedError(
            "Only L-inf, L1 and L2 norms are currently implemented."
        )

    # Scale perturbation to be the solution for the norm=eps rather than norm=1 problem
    scaled_perturbation = tf.multiply(eps, optimal_perturbation)
    return scaled_perturbation


def fgsm_attack(
    X_test, 
    y_test, 
    model, 
    epsilon, 
    norm,
    clip_min = None,
    clip_max = None):

    inp = tf.convert_to_tensor(X_test, dtype = tf.float32)
    imgv = tf.Variable(inp)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(imgv)
        predictions = model(imgv)
        loss = tf.keras.losses.MeanSquaredError()(y_test, predictions)
        grads = tape.gradient(loss,imgv)

    perturbation = optimize_linear(grads, epsilon, norm)
    inp = inp + perturbation

    if clip_min is not None or clip_max is not None:
        inp = tf.clip_by_value(inp, clip_min, clip_max)
    # signed_grads = tf.sign(grads)
    # inp = inp + (epsilon*signed_grads)

    return inp

def pgd_attack(
    X_test,
    y_test,
    model,
    iterations,
    alpha,
    epsilon,
    norm,
    clip_min = None,
    clip_max = None):
    
    gen_img = tf.convert_to_tensor(X_test, dtype=tf.float32)
    gen_img = tf.clip_by_value(gen_img, X_test-epsilon, X_test+epsilon)
    eta = tf.zeros_like(gen_img)
    gen_img = clip_eta(eta, norm, epsilon)
    x_temp = X_test

    for iters in range(iterations):
        imgv = tf.Variable(gen_img)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(imgv)
            predictions = model(imgv)
            loss = tf.keras.losses.MeanSquaredError()(y_test, predictions)
            grads = tape.gradient(loss,imgv)

        perturbation = optimize_linear(grads, alpha, norm)
        gen_img = gen_img + perturbation
        # signed_grads = tf.sign(grads)
        # gen_img = gen_img + (alpha*signed_grads)

        eta = gen_img - tf.convert_to_tensor(X_test, dtype=tf.float32)
        eta = clip_eta(eta, norm, epsilon)
        gen_img = tf.convert_to_tensor(X_test, dtype=tf.float32) + eta
        # gen_img = tf.clip_by_value(gen_img, X_test-epsilon, X_test+epsilon)

        if clip_min is not None or clip_max is not None:
            gen_img = tf.clip_by_value(gen_img, clip_min, clip_max)

        
    return gen_img