# IMPORT USEFUL LIBRARIES:

import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf

# RANDOM MINI BATCHES:

def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        k = num_complete_minibatches
        
        mini_batch_X = shuffled_X[:, k * mini_batch_size :]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

# DEFINE FUNCTION TO SPLIT THE DATA:

def data_division(X, Y):
    """
    

    Parameters
    ----------
    X : 2-D Array of size: n_x, m
        where n_x is the number of input variables and m are the training examples.
        It will be divided to two 2-D arrays: X_train and X_test.
        
    Y : 2-D Array of size: 1, m
        where m are the training examples.
        It will be divided to two 2-D arrays: Y_train and Y_test.

    Returns
    -------
    Three dictionaries: Train, Dev, Test, where:
        Train contains the matrices: X_train and Y_train (keys: X_train, Y_train)
        Train contains the matrices: X_dev and Y_dev (keys: X_dev, Y_dev)
        Test contains the matrices: X_test and Y_test (keys: X_test, Y_test)

    """
    
    Train = dict()                # Initialize Train dictionary.
    Dev = dict()                  # Initialize Development dictionary.
    Test = dict()                 # Initialize Test dictionary.
    
    n_x, m = X.shape              # Get the geometry of the problem.
    
    if (m < 500 * 10**3):         # Split the dataset.
        per_test = 20 / 10**2     # For small datasets 20% goes to testing.
        per_dev = 20 / 10**2      # For small datasets 20% goes to development.
    elif (m < 1.5 * 10**6):
        per_test = 1 / 10**2      # For big datasets 1% goes to testing.
        per_dev = 1 / 10**2      # For big datasets 1% goes to development.
    else:
        per_test = 1 / 10**3      # For very big datasets 0.1% goes to testing.
        per_dev = 4 / 10**3      # For very big datasets 0.4% goes to development.
    
    m_test = np.floor(per_test * m)
    m_test = int(m_test)          # Slice indices must be integers.
    m_dev = np.floor(per_dev * m)
    m_dev = int(m_dev)          # Slice indices must be integers.
    m_train = m - (m_dev + m_test)
    
    X_train = X[:, :m_train]
    Y_train = Y[:, :m_train]
    
    X_dev = X[:, m_train:(m_train + m_dev)]
    Y_dev = Y[:, m_train:(m_train + m_dev)]
    
    X_test = X[:, (m_train + m_dev):]
    Y_test = Y[:, (m_train + m_dev):]
    
    Train = {'X': X_train, 'Y': Y_train}
    Dev = {'X': X_dev, 'Y': Y_dev}
    Test = {'X': X_test, 'Y': Y_test}
    
    return Train, Dev, Test

# IMPORT X,Y AS PLACEHOLDERS:

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "tf.float32"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "tf.float32"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    X = tf.placeholder(tf.float32, [n_x, None], name = 'X')
    Y = tf.placeholder(tf.float32, [n_y, None], name = 'Y')
    
    return X, Y

# INITIALIZATION:

def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [6, 2]
                        b1 : [6, 1]
                        W2 : [4, 6]
                        b2 : [4, 1]
                        W3 : [1, 4]
                        b3 : [1, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
        
    W1 = tf.get_variable("W1", [6, 2], initializer = tf.initializers.he_normal())
    b1 = tf.get_variable("b1", [6, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [4, 6], initializer = tf.initializers.he_normal())
    b2 = tf.get_variable("b2", [4, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1, 4], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [1, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

# FORWARD PROPAGATION:

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
                                                           # Numpy Equivalents:
    Z1 = tf.matmul(W1, X) + b1                             # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2                            # Z2 = np.dot(W2, A1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3                            # Z3 = np.dot(W3, A2) + b3
    
    return Z3

# COST FUNCTION:

def compute_cost(Z3, Y, parameters, lambd):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (1, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    lambd -- L2 regularization parameter
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels)
    cost = tf.reduce_mean(cost)
    cost += tf.add_n([ tf.nn.l2_loss(W) for W in [W1, W2, W3] ]) * lambd
    
    return cost

# MODEL:

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, lambd = 0.01,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 2, number of training examples = )
    Y_train -- training set, of shape (output size = 1, number of training examples = )
    X_test -- test set, of shape (input size = 1, number of training examples = )
    Y_test -- test set, of shape (output size = 1, number of test examples = )
    learning_rate -- learning rate of the optimization
    lambd -- L2 regularization parameter
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
#     ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = list()                                    # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y, parameters, lambd)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feed_dict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += minibatch_cost / minibatch_size

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate) + "\n" + "Lambda =" + str(lambd))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        Y_hat = tf.sigmoid(Z3)
        Y_hat = tf.round(Y_hat)                    # banker's rounding..
        correct_prediction = tf.equal(Y_hat, Y)

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters
