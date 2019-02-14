#%%
import numpy as np
import scipy
import matplotlib.pyplot as plt

%load_ext autoreload
%autoreload 2
from dnn_app_utils import *
from file_utils import *
from cs231n.data_utils import load_CIFAR10

# %%

# Only use for Colab
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null

from google.colab import drive
drive.mount('/content/drive')

# Only use for Colab
!rm -r datasets 
#!mkdir datasets
!cp -a '/content/drive/My Drive/Colab Notebooks/datasets/' '/content/'
#%%
# Load data
train_x_orig, train_y, test_x_orig, test_y = load_data()

# %%
# Shuffle data, and use one small part
m_train_take = 25000
m_test_take = 10000
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
layers_dims = [num_px*num_px*3, 3, 3, 10] # Use small model for checking
parameters = initialize_parameters_deep(layers_dims)
AL, caches = L_model_forward(train_x[:,0:m_testGrad_take], parameters)
grads = L_model_backward(AL, train_y[:,0:m_testGrad_take], caches, lambd = 0,keep_prob = 1)
gradient_check_n(parameters, grads, layers_dims, train_x[:,0:m_testGrad_take], train_y[:,0:m_testGrad_take])

#%%
### layer model ###
layers_dims = [num_px*num_px, 30, 50, 30, 10] #  X-layer model
#%%
parameters = L_layer_model(train_x, train_y, layers_dims, optimizer = "momentum", learning_rate = 0.03, num_epochs = 100, print_cost = True, mini_batch_size = 128)
predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)


#%%
filename = 'parameters.h5'
save_dict_to_hdf5(parameters, filename)

#%%
parameters = load_dict_from_hdf5('parameters.h5')
predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)

#%%

#%%
# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
train_x_orig = X_train
train_y = y_train
test_x_orig = X_test
test_y = y_test


#%%
# Shuffle data, and use one small part
m_train_take = 25000
m_test_take = 10000
permutation = list(np.random.permutation(train_x_orig.shape[0]))
train_x_orig = train_x_orig[permutation, :]
train_y = train_y[permutation]

train_x_orig = train_x_orig[0:m_train_take, :]
train_y = train_y[0:m_train_take]

test_x_orig = test_x_orig[0:m_test_take, :]
test_y = test_y[0:m_test_take]
# %%

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

train_x_flatten = train_x_orig.reshape(m_train, num_px*num_px*3).T
test_x_flatten = test_x_orig.reshape(m_test, num_px*num_px*3).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

# train_x = train_x_flatten
# test_x = test_x_flatten
#%%
# X = train_x.reshape(m_train, 32, 32, 3)
X = train_x.reshape(m_train, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

#%%
index = 15
plt.imshow(X[index])

#%%
ig, axes1 = plt.subplots(5,5,figsize=(3,3))
for j in range(5):
    for k in range(5):
        i = np.random.choice(range(len(X)))
        axes1[j][k].set_axis_off()
        axes1[j][k].imshow(X[i:i+1][0])


#%%
import sys
sys.modules[__name__].__dict__.clear()

#%%
