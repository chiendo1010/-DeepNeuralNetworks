#%%
import numpy as np
import scipy
import matplotlib.pyplot as plt

from dnn_app_utils import *

%load_ext autoreload
%autoreload 2

#%%
# Load data
train_x_orig, train_y, test_x_orig, test_y = load_data()

# %%
# Shuffle data, and use one small part
m_train = 5000
m_test = 500
permutation = list(np.random.permutation(train_x_orig.shape[0]))
train_x_orig = train_x_orig[permutation, :]
train_y = train_y[permutation]

train_x_orig = train_x_orig[0:m_train, :]
train_y = train_y[0:m_train]

test_x_orig = test_x_orig[0:m_test, :]
test_y = test_y[0:m_test]

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

#%%
### CONSTANTS ###
layers_dims = [num_px*num_px, 20, 7, 5, 10] #  4-layer model

#%%
# GRADED FUNCTION: L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075*3*2, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

#%%
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
predictions_train = predict(train_x, train_y, parameters)
#%%
predictions_test = predict(test_x, test_y, parameters)

#%%
Hello()

#%%


#%%
