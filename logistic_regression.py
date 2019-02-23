'''
Filename: logistic_regression.py
Path: logistic_regression_digits
Created Date: Saturday, February 23rd 2019, 10:19:51 am
Author: kelson martins
'''


from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import math

# Sigmoid function
def logistic(x):
	
	x = 1 / (1 + np.exp(-x))

	return x


def hypothesisLogistic(X, coefficients, bias):
    
    yPred = logistic(np.dot(X,coefficients.T) + bias)

    return yPred


def gradient_descent_log(bias, coefficients, alpha, X, Y, max_iter):

    length = len(Y)
    
    # array is used to store change in cost function for each iteration of GD
    errorValues = []
    
    for num in range(0, max_iter):
        
        # Calculate predicted y values for current coefficient and bias values 
        predictedY = hypothesisLogistic(X, coefficients, bias)
        
        # calculate gradient for bias
        biasGrad =    (1.0/length) *  (np.sum( predictedY - Y))
        
        #update bias using GD update rule
        bias = bias - ( alpha * biasGrad )
        
        # for loop to update each coefficient value in turn
        for coefNum in range(len(coefficients)):
            
            # calculate the gradient of the coefficient
            gradCoef = (1.0/length)* (np.sum( (predictedY - Y)*X[:, coefNum]))
            
            # update coefficient using GD update rule
            coefficients[coefNum] = coefficients[coefNum] - ( alpha * gradCoef ) 

        # Cross Entropy Error 
        cost = 1/length * np.sum( -(Y*np.log(predictedY)) - (1-Y) * np.log(1-predictedY))

        errorValues.append(cost)

    
    # plot the cost for each iteration of gradient descent
    plt.plot(errorValues)
    plt.show()
    
    return bias, coefficients

def calculateAccuracy(bias, coefficients, X_test, y_test):

	length = X_test.shape[0]
	yPred = hypothesisLogistic(X_test,coefficients,bias)

	yPred[yPred >= 0.5] = 1
	yPred[yPred < 0.5] = 0

	comparison = yPred == y_test
	correct_predicted = np.sum(comparison)

	accuracy = correct_predicted / length


	print ("Final Accuracy on Test Set: {}".format(accuracy))
    

def logisticRegression(X_train, y_train, X_test, y_test):

    # set the number of coefficients equal to the number of features
    coefficients = np.zeros(X_train.shape[1])
    bias = 0.0
   
    alpha = 0.001 # learning rate
    
    max_iter=200

    # call gredient decent, and get intercept(bias) and coefficents
    bias, coefficients = gradient_descent_log(bias, coefficients, alpha, X_train, y_train, max_iter)
    
    calculateAccuracy(bias, coefficients, X_test, y_test)
    
def main():
    
    digits = datasets.load_digits()
    
    # Display one of the images to the screen
    # plt.figure(1, figsize=(3, 3))
    # plt.imshow(digits.images[3], cmap=plt.cm.gray_r, interpolation='nearest')
    # plt.show()
    
    # Load the feature data and the class labels
    X_digits = digits.data
    y_digits = digits.target

    print(digits.data)
    
    # The logistic regression model will differentiate between two digits
    # Code allows you specify the two digits and extract the images 
    # related to these digits from the dataset
    indexD1 = y_digits==1
    indexD2 = y_digits==7
    allindices = indexD1 | indexD2
    X_digits = X_digits[allindices]
    y_digits = y_digits[allindices]
 
    # We need to make sure that we convert the labels to 
    # 0 and 1 otherwise our cross entropy won't work 
    lb = preprocessing.LabelBinarizer()
    y_digits = lb.fit_transform(y_digits)
    y_digits = y_digits.flatten()

    n_samples = len(X_digits)

    # Seperate data in training and test
    # Training data 
    X_train = X_digits[:int(.7 * n_samples)]
    y_train = y_digits[:int(.7 * n_samples)]
    
    # Test data
    X_test = X_digits[int(.7 * n_samples):]
    y_test = y_digits[int(.7 * n_samples):]

    logisticRegression(X_train, y_train, X_test, y_test)  

main()
