# IMPORT USEFUL LIBRARIES:

import numpy as np
import matplotlib.pyplot as plt
import csv
from dnn_func import data_division, model



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

Train, Dev, Test = data_division(X, Y)

X_train = Train['X']
Y_train = Train['Y']

X_dev = Dev['X']
Y_dev = Dev['Y']

X_test = Test['X']
Y_test = Test['Y']

# TRAINING SET SIZE:

m = X_train.shape[1]                                                # number of training examples
n_x = X_train.shape[0]                                              # number of input variables
# print(n_x, m)

# TRAIN THE DEEP NEURAL NETWORK:

plt.figure(1)

parameters = model(X_train, Y_train, X_dev, Y_dev, lambd = 0.01, num_epochs = 1500, minibatch_size = 256)