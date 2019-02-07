#%%
import numpy as np
import scipy
import matplotlib.pyplot as plt

%load_ext autoreload
%autoreload 2
from dnn_app_utils import *

#%%
# Load data
train_x_orig, train_y, test_x_orig, test_y = load_data()

# %%
# Shuffle data, and use one small part
m_train_take = 5000
m_test_take = 500
permutation = list(np.random.permutation(train_x_orig.shape[0]))
train_x_orig = train_x_orig[permutation, :]
train_y = train_y[permutation]

train_x_orig = train_x_orig[0:m_train_take, :]
train_y = train_y[0:m_train_take]

test_x_orig = test_x_orig[0:m_test_take, :]
test_y = test_y[0:m_test_take]

#%%
# Plot
index = 15
plt.imshow(train_x_orig[index])
print("y = " + str(train_y[index]))


#%%
# Explore your dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

train_y = train_y.reshape(1,-1)
test_y = test_y.reshape(1,-1)

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ")")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

train_x_flatten = train_x_orig.reshape(m_train, num_px*num_px).T
test_x_flatten = test_x_orig.reshape(m_test, num_px*num_px).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

#%%
#  Gradient Checking
m_testGrad_take = 10
layers_dims = [num_px*num_px, 3, 3, 10] # Use small model for checking
parameters = initialize_parameters_deep(layers_dims)
AL, caches = L_model_forward(train_x[:,0:m_testGrad_take], parameters)
grads = L_model_backward(AL, train_y[:,0:m_testGrad_take], caches)
gradient_check_n(parameters, grads, layers_dims, train_x[:,0:m_testGrad_take], train_y[:,0:m_testGrad_take])

#%%
### CONSTANTS ###
layers_dims = [num_px*num_px, 20, 7, 5, 10] #  4-layer model
parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate = 0.09, num_iterations = 1500, print_cost = True)
predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)
playSoundFinish()

#%%


#%%
