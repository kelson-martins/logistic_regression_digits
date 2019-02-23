### Python Logistic Regression

This python program implements a logistic regression algorithm that identifies data between 2 handwritten digits from the [handwritten digits dataset](http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits).

#### Activation Function
Activation function used is the Sigmoid.

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;h(X)&space;=&space;\frac{\mathrm{1}&space;}{\mathrm{1}&space;&plus;&space;e^-&space;X\alpha^T&plus;b&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;h(X)&space;=&space;\frac{\mathrm{1}&space;}{\mathrm{1}&space;&plus;&space;e^-&space;X\alpha^T&plus;b&space;}" title="h(X) = \frac{\mathrm{1} }{\mathrm{1} + e^- X\alpha^T+b }" /></a>

### Loss Function
Cross Entropy Loss Function is being used.

### Trainning and Test Data
By default:

70% of the dataset is being used for training

30% of the dataset is being used for testing

