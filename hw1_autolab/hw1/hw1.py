import numpy as np
import copy

def load_data(file):
    table = np.genfromtxt(file, delimiter=',')
    return table[:, :784], table[:, 784].astype(int)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def deriv_sigmoid(activation, *args):
    return activation * (1 - activation)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def deriv_softmax(YHat, Y, batch_size):
    return (YHat - Y) / batch_size

def tanh(x):
    return np.tanh(x)

def deriv_tanh(x, *args):
    return 1.0 - np.tanh(x) ** 2

def relu(x):
    np.maximum(x, 0, x)
    return x

def deriv_relu(x, *args):
    return (x > 0) * 1.0

# def cross_entroy_loss(y, t):
#     return - np.sum(t * np.log(y) + (1-t) * np.log(1-y))

def onehot(y, cols=10):
    rows = y.shape[0]
    o = np.zeros((rows, cols))
    o[np.arange(rows), y] = 1
    return o

def update_weights_perceptron(X, Y, inputWeights, inputBias, lr):
    batch_size = X.shape[0]

    w = inputWeights[0]
    b = inputBias[0]

    Y = onehot(Y)  # 3000x10

    # dot
    Z = np.dot(X, w) + b  # 3000x10

    YHat = softmax(Z)

    dEdZ = (YHat - Y) / batch_size
    dZdw = X

    grad_w = np.dot(dZdw.T, dEdZ)
    grad_b = np.sum(dEdZ, axis=0)

    w = w - lr * grad_w
    b = b - lr * grad_b

    return [w], [b]


def update_weights_single_layer(X, Y, inputWeights, inputBias, lr):
    batch_size = X.shape[0]
    N = 1 + 2

    Y = onehot(Y)  # 3000x10

    activations = [None] * N
    activations[0] = X

    ac_funs = [None, sigmoid, softmax]

    for layer in range(1, N):
        # dot
        z = np.dot(activations[layer - 1], inputWeights[layer - 1]) + inputBias[layer - 1]  # 3000x10
        # activation: softmax
        activations[layer] = ac_funs[layer](z) # 3000x10

    YHat = activations[-1]

    grads_z1 = (YHat - Y) / batch_size
    grads_w1 = np.dot(activations[1].T, grads_z1)
    grads_b1 = np.sum(grads_z1, axis=0)
    grads_a1 = np.dot(grads_z1, inputWeights[1].T)


    grads_z0 = activations[1] * (1 - activations[1]) * grads_a1
    grads_w0 = np.dot(activations[0].T, grads_z0)
    grads_b0 = np.sum(grads_z0, axis=0)
    # grads_a0 = np.dot(grads_z1, inputWeights[1].T)

    inputWeights[1] -= lr * grads_w1
    inputBias[1] -= lr * grads_b1

    inputWeights[0] -= lr * grads_w0
    inputBias[0] -= lr * grads_b0

    return inputWeights, inputBias

def update_weights_double_layer(X, Y, inputWeights, inputBias, lr, activation_fun_name="sigmoid",
                                momentum=0.0, lastDeltaW=None, lastDeltaB=None):
    batch_size = X.shape[0]
    N = 2 + 2

    Y = onehot(Y)  # 3000x10

    activations = [None] * N
    activations[0] = X

    activation_fun_dict = {
        "sigmoid" : (sigmoid, deriv_sigmoid),
        "tanh"    : (tanh, deriv_tanh),
        "relu"    : (relu, deriv_relu)
    }
    activation_fun = activation_fun_dict[activation_fun_name][0]
    deriv_activation_fun = activation_fun_dict[activation_fun_name][1]

    ac_funs = [None, activation_fun, activation_fun, softmax]

    # forward pass
    for layer in range(1, N):
        # dot
        z = np.dot(activations[layer - 1], inputWeights[layer - 1]) + inputBias[layer - 1]  # 3000x10
        # activation
        activations[layer] = ac_funs[layer](z) # 3000x10

    # z = x * w + b, where x is the activation from previous layer
    # a = activation_fun(z)
    grads_z = [None] * N  # dE/dz
    grads_w = [None] * N  # dE/dw
    grads_b = [None] * N  # dE/db
    grads_a = [None] * N  # dE/da
    grads_a[-1] = 1 # special grad for backprop softmax. Already done da/dz * dE/da in deriv_softmax

    deriv_ac_funs = [None, deriv_activation_fun, deriv_activation_fun, deriv_softmax]

    # backward pass
    for layer in range(N - 1, 0, -1):
        grads_z[layer - 1] = deriv_ac_funs[layer](activations[layer], Y, batch_size) * grads_a[layer]
        grads_w[layer - 1] = np.dot(activations[layer - 1].T, grads_z[layer - 1])
        grads_b[layer - 1] = np.sum(grads_z[layer - 1], axis=0, keepdims=True)
        grads_a[layer - 1] = np.dot(grads_z[layer - 1], inputWeights[layer - 1].T)

    # param update
    for layer in range(N - 1, 0, -1):
        deltaW = -lr * grads_w[layer - 1]
        deltaB = -lr * grads_b[layer - 1]

        # momentum
        if momentum != 0.0:
            deltaW += lastDeltaW[layer - 1] * momentum
            deltaB += lastDeltaB[layer - 1] * momentum
            lastDeltaW[layer - 1], lastDeltaB[layer - 1] = map(copy.deepcopy, (deltaW, deltaB))

        inputWeights[layer - 1] += deltaW
        inputBias[layer - 1] += deltaB

    return inputWeights, inputBias

def update_weights_double_layer_act(X, Y, inputWeights, inputBias, lr, fun_name):
    return update_weights_double_layer(X, Y, inputWeights, inputBias, lr, activation_fun_name=fun_name)


def update_weights_double_layer_act_mom(X, Y, inputWeights, inputBias, lr, fun_name, momentum, epochs):
    lastDeltaW = [w - w for w in inputWeights]
    lastDeltaB = [b - b for b in inputBias]
    for epoch in range(epochs):
        print("epoch", epoch)
        update_weights_double_layer(X, Y, inputWeights, inputBias, lr=lr, activation_fun_name=fun_name,
                                    momentum=momentum, lastDeltaW=lastDeltaW, lastDeltaB=lastDeltaB)
    return inputWeights, inputBias

