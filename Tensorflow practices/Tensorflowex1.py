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

# %% increasing dataset of the training set szie using//// np.tile numpy.tile(A, reps)
Xt = np.tile(Xn,(1000,1))
Yt= np.tile(Y,(1000,1))   
print(Xt.shape, Yt.shape)  
# %% TF Model
#in order to initialize same random values (in tf weighrs/paras)
tf.random.set_seed(1234)
model = Sequential(
    [
        tf.keras.Input(shape = (2,)),
        Dense(3, activation ='sigmoid', name = 'Layer1'),
        Dense(1, activation ='sigmoid', name = 'Layer2')
    ]
)
model.summary()
# %%weights and Biases
W1, b1 = model.get_layer('Layer1').get_weights()
W2, b2 = model.get_layer('Layer2').get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W1{W2.shape}:\n", W2, f"\nb1{b2.shape}:", b2)
# %%
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
)
model.fit(
    Xt,Yt, epochs=10
)
# %%Updated weights
W1, b1 = model.get_layer("Layer1").get_weights()
W2, b2 = model.get_layer("Layer2").get_weights()
print(f'W1:\n', W1, "\n b1:", b1)
print(f'W2:\n', W2, "\n b2:", b2)

# %%
