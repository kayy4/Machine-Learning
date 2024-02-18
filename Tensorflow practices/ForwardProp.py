#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# %%
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
# %%This dataset is a coffe roasting dataset from Andrew Ng 
def dataset_creation():
    rndm = np.random.default_rng(2)
    X = rndm.random(400).reshape(-1, 2)  #array of shape (200, 2)
    #In this dataset the second collomn is a duration of time which GOOD DAta is range 12-15
    X[:,1] = X[:,1]*4 + 11.5
    #In the first column its temperature range of 175-260°C
    X[:,0] = X[:, 0] * (285 - 150) + 150
    
    Y = np.zeros(len(X)) #for labels array of 0s for the length of X (200)
    i=0
    for t,d in X:
        y = -3/(260-175)*t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d<=y ):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1
    
    return (X, Y.reshape(-1, 1))  #Y is aloso reshaped into an 2d array
# %%
X,Y = dataset_creation()
print (X.shape, Y.shape)

# %%Normalizing data // scaling the data so that the features have similar ranges or distributions
print (f'X1 MAX, MIN Pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}')
print (f'X2 MAX, MIN Pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}')

normlization = tf.keras.layers.Normalization(axis= -1)
normlization.adapt(X)
Xn = normlization(X)

print (f'X1 MAX, MIN After normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}')
print (f'X2 MAX, MIN After normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}')

# %%
# Define the activation function
#from tensorflow.math import sigmoid
# or we can 
#implementing sigmoid function from scratch for fun
def sigmoid(z):
    z = np.clip( z, -500, 500 )           # protect against overflow
    g = 1.0/(1.0+np.exp(-z))			# = 1/1+e^(−z)
    return g

g = sigmoid

# %%DEnse layer from scratch 
def my_dense(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:,j]
        z = np.dot(w, a_in) +b[j]
        a_out[j] = g(z)
    return (a_out)
# %%foward prop frm scratch (For 2 layer network)
def my_sequential(x, W1, b1, W2,b2):
    a1 = my_dense(x, W1, b1)
    a2 = my_dense(a1, W2, b2)
    return(a2)

# %% some dummy weight s n biases to test
W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )
# %%
def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0]= my_sequential(X[i], W1, b1, W2, b2)
    return(p)
# %% test data 
X_tst = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_tstn = normlization(X_tst)  # remember to normalize
predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)
print('Predictions = \n',predictions )
# %%
