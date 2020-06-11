import numpy as np
import sys
from datetime import datetime
from numpy import genfromtxt

start=datetime.now()


n = len(sys.argv)
if n==4:
    x_train = sys.argv[1]
    y_train = sys.argv[2]
    x_test = sys.argv[3]
    x_train = genfromtxt(x_train, delimiter=',')
    x_test = genfromtxt(x_test, delimiter=',')
    y_train = genfromtxt(y_train, delimiter=',')
else:
    x_train = genfromtxt('train_image.csv', delimiter=',')
    x_test = genfromtxt('test_image.csv', delimiter=',')
    y_train = genfromtxt('train_label.csv', delimiter=',')


#getting the test and training set ready

#x_train = genfromtxt('train_image.csv', delimiter=',')
#x_test = genfromtxt('test_image.csv', delimiter=',')
#y_train = genfromtxt('train_label.csv', delimiter=',')
#y_test = genfromtxt('test_label.csv', delimiter=',')

#x_test = genfromtxt('train_image.csv', delimiter=',')
#x_train = genfromtxt('test_image.csv', delimiter=',')
#y_test = genfromtxt('train_label.csv', delimiter=',')
#y_train = genfromtxt('test_label.csv', delimiter=',')

x_train = x_train/255
x_test = x_test/255

digits = 10

m = x_train.shape[0]
m_test = x_test.shape[0]

y_train = y_train.reshape(1,m)
#y_test = y_test.reshape(1,m_test)

y_new_train = np.eye(digits)[y_train.astype('int32')]
y_new_train = y_new_train.T.reshape(10,m)


#y_new_test = np.eye(digits)[y_test.astype('int32')]
#y_new_test = y_new_test.T.reshape(10,m_test)


x_train = x_train.T
x_test = x_test.T


#sigmoid function
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

#cross entropy loss
def compute_loss(X, Y):

    L_sum = np.sum(np.multiply(X, np.log(Y)))
    m = X.shape[1]
    L = -(1/m) * L_sum

    return L

def softmax(z):
    s = np.exp(z) / np.sum(np.exp(z), axis=0)
    return s

#forward pass
def feed_forward(X, parameters):

    cache = {}
    #Z1 = W1*X+b1
    cache["Z1"] = np.matmul(parameters["W1"], X) + parameters["b1"]
    #A1 = sigmoid(Z1)
    cache["A1"] = sigmoid(cache["Z1"])
    #Z2 = W2*A1+b2
    cache["Z2"] = np.matmul(parameters["W2"], cache["A1"]) + parameters["b2"]
    #A2 = sigmoid(Z2)
    cache["A2"] = sigmoid(cache["Z2"])
    #Z3 = W3*A2+b3
    cache["Z3"] = np.matmul(parameters["W3"], cache["A2"]) + parameters["b3"]
    #A3 = softmax(Z3)
    cache["A3"] = softmax(cache["Z3"])

    return cache

def back_propagate(X, Y, parameters, cache):
    #find differentiation of all the parameters and update the gradients and store in the dictionary

    #softmax cost differentiation eventually boils down to difference of the terms 
    dZ3 = cache["A3"] - Y
    dW3 = (1./m_batch) * np.matmul(dZ3, cache["A2"].T)
    db3 = (1./m_batch) * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.matmul(parameters["W3"].T, dZ3)
    dZ2 = dA2 * sigmoid(cache["Z2"]) * (1 - sigmoid(cache["Z2"]))
    dW2 = (1./m_batch) * np.matmul(dZ2, cache["A1"].T)
    db2 = (1./m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(parameters["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))
    dW1 = (1./m_batch) * np.matmul(dZ1, X.T)
    db1 = (1./m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    gradient = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}

    return gradient

np.random.seed(138)


X = x_train
Y = y_new_train

#hyperparameters
n_x = x_train.shape[0]
n_h1 = 512
n_h2 = 64
learning_rate = 4
beta = .9
batch_size = 64
batches = -(-m // batch_size)
epochs = 20

# initialization
parameters = { "W1": np.random.randn(n_h1, n_x) * np.sqrt(1. / n_x),
           "b1": np.zeros((n_h1, 1)) * np.sqrt(1. / n_x),
           "W2": np.random.randn(n_h2, n_h1) * np.sqrt(1. / n_h1),
           "b2": np.zeros((n_h2, 1)) * np.sqrt(1. / n_h1),
           "W3": np.random.randn(digits, n_h2) * np.sqrt(1. / n_h2),
           "b3": np.zeros((digits, 1)) * np.sqrt(1. / n_h2)}

V_dW1 = np.zeros(parameters["W1"].shape)
V_db1 = np.zeros(parameters["b1"].shape)
V_dW2 = np.zeros(parameters["W2"].shape)
V_db2 = np.zeros(parameters["b2"].shape)
V_dW3 = np.zeros(parameters["W3"].shape)
V_db3 = np.zeros(parameters["b3"].shape)


# training the model
for i in range(epochs):

    #using mini batch gradient descent
    permutation = np.random.permutation(x_train.shape[1])
    X_train_shuffled = x_train[:, permutation]
    Y_train_shuffled = y_new_train[:, permutation]

    for j in range(batches):

        begin = j * batch_size
        end = min(begin + batch_size, x_train.shape[1] - 1)
        X = X_train_shuffled[:, begin:end]
        Y = Y_train_shuffled[:, begin:end]
        m_batch = end - begin

        cache = feed_forward(X, parameters)
        gradient = back_propagate(X, Y, parameters, cache)

        V_dW1 = (beta * V_dW1 + (1. - beta) * gradient["dW1"])
        V_db1 = (beta * V_db1 + (1. - beta) * gradient["db1"])
        V_dW2 = (beta * V_dW2 + (1. - beta) * gradient["dW2"])
        V_db2 = (beta * V_db2 + (1. - beta) * gradient["db2"])
        V_dW3 = (beta * V_dW3 + (1. - beta) * gradient["dW3"])
        V_db3 = (beta * V_db3 + (1. - beta) * gradient["db3"])

        parameters["W1"] = parameters["W1"] - learning_rate * V_dW1
        parameters["b1"] = parameters["b1"] - learning_rate * V_db1
        parameters["W2"] = parameters["W2"] - learning_rate * V_dW2
        parameters["b2"] = parameters["b2"] - learning_rate * V_db2
        parameters["W3"] = parameters["W3"] - learning_rate * V_dW3
        parameters["b3"] = parameters["b3"] - learning_rate * V_db3

    cache = feed_forward(x_train, parameters)
    train_cost = compute_loss(y_new_train, cache["A3"])
    #print("Epoch {}: training cost = {}".format(i+1 ,train_cost))

#print("Done.")

#prediction
cache = feed_forward(x_test, parameters)
predictions = np.argmax(cache["A3"], axis=0)

#labels = np.argmax(y_new_test, axis=0)

#print(classification_report(predictions, labels))

#print(confusion_matrix(predictions, labels))
#print(classification_report(predictions, labels))

#correct = 0
#for i in range(len(predictions)):
#    if predictions[i] == labels[i]:
#        correct = correct + 1
#print(correct)

#saving predictions to test_predictions.csv
np.savetxt("test_predictions.csv", predictions, delimiter=",", fmt="%d")

endtime = datetime.now()
print('time taken =', endtime - start)

