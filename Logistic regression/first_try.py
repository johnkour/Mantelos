# IMPORTING USEFULL LIBRARIES:

import numpy as np
import matplotlib.pyplot as plt
import csv
import sklearn
import sklearn.datasets
import sklearn.linear_model

# DEFINE DECISION BOUNDARY FUNCTION:

# The following function was provided by:
# https://github.com/rvarun7777/Deep_Learning/blob/master/Neural%20Networks%20and%20Deep%20Learning/Week%203/Planar%20data%20classification%20with%20one%20hidden%20layer/planar_utils.py

def plot_decision_boundary(model, X, y):             
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    # plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)

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
    Two dictionaries: Train, Test, where:
        Train contains the matrices: X_train and Y_train (keys: X_train, Y_train)
        Test contains the matrices: X_test and Y_test (keys: X_test, Y_test)

    """
    
    Train = dict()                # Initialize Train dictionary.
    Test = dict()                 # Initialize Test dictionary.
    
    n_x, m = X.shape              # Get the geometry of the problem.
    
    if (m < 500 * 10**3):         # Split the dataset.
        per_test = 20 / 10**2     # For small datasets 20% goes to testing.
    elif (m < 1.5 * 10**6):
        per_test = 2 / 10**2      # For big datasets 2% goes to testing.
    else:
        per_test = 5 / 10**3      # For very big datasets 20% goes to testing.
    
    m_test = np.floor(per_test * m)
    m_test = int(m_test)          # Slice indices must be integers.
    m_train = m - m_test
    
    X_train = X[:, :m_train]
    Y_train = Y[:, :m_train]
    
    X_test = X[:, m_train:]
    Y_test = Y[:, m_train:]
    
    Train = {'X': X_train, 'Y': Y_train}
    Test = {'X': X_test, 'Y': Y_test}
    
    return Train, Test

# IMPORT DATA:

results = list()

f_name = input('Please enter which file you would like to access: (for Data_short press enter)' + '\n')
if (len(f_name) < 1):    f_name = 'Data_short'
f_name += '.csv'

with open(f_name) as csvfile: 
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader:                                         # each row is a list
        results.append(row)

Data = np.array(results)
# print(Data)

X = Data[:, :-1]
X = X.T
Y = Data[:, -1]
Y = Y.reshape(1, len(Y))
# print(X.shape)
# print(Y.shape)

# PLOT THE DATASET:

plt.figure(0)
plt.scatter(X[0, :], X[1, :], c=Y.reshape(Y.shape[1],), s = 40, cmap=plt.cm.Spectral)

# DIVIDE THE DATA INTO TRAIN AND TEST SETS:

Train, Test = data_division(X, Y)

X_train = Train['X']
Y_train = Train['Y']

X_test = Test['X']
Y_test = Test['Y']

# TRAINING SET SIZE:

m = X_train.shape[1]                                                # number of training examples
n_x = X_train.shape[0]                                              # number of input variables
# print(n_x, m)

# TRAIN THE LOGISTIC REGRESSION CLASSIFIER:

clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X_train.T, np.ravel(Y_train));

# PLOT LOGISTIC REGRESSION DECISION BOUNDARY:

plt.figure(1)
plot_decision_boundary(lambda x: clf.predict(x), X_train, np.ravel(Y_train))
plt.scatter(X_train[0, :], X_train[1, :], c=Y_train.reshape(Y_train.shape[1],), cmap=plt.cm.Spectral)
plt.title("Logistic Regression in training")

# MAKE PREDICTIONS FOR TRAINING:

Accur = dict()

Y_hat = clf.predict(X_train.T)
Accur['Train'] = (np.dot(Y_train,Y_hat) + np.dot(1-Y_train,1-Y_hat))/(Y_train.size)

print ('Accuracy of logistic regression in training: %d ' % float((np.dot(Y_train,Y_hat) + np.dot(1-Y_train,1-Y_hat))/float(Y_train.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

# MAKE PREDICTIONS FOR TESTING:

Y_hat = clf.predict(X_test.T)
Accur['Test'] = (np.dot(Y_test,Y_hat) + np.dot(1-Y_test,1-Y_hat))/(Y_test.size)

print ('Accuracy of logistic regression in testing: %d ' % float((np.dot(Y_test,Y_hat) + np.dot(1-Y_test,1-Y_hat))/float(Y_test.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

# PLOT LOGISTIC REGRESSION DECISION BOUNDARY:

plt.figure(2)
plot_decision_boundary(lambda x: clf.predict(x), X_train, np.ravel(Y_train))
plt.scatter(X_test[0, :], X_test[1, :], c=Y_test.reshape(Y_test.shape[1],), cmap=plt.cm.Spectral)
plt.title("Logistic Regression in testing")
