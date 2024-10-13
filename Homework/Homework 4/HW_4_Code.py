#ISE 364/464 Homework 4 - Problem 4
#Implementation of training a Logistic Regression Model via gradient descent
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression


###############################################################################
######################### Dataset Generation Functions ########################
###############################################################################

def Set_Seed(seed):
    """
    Simple function to sed the random seed for the relevant libraries that one will be using.

    Parameters
    ----------
    seed : int
        The value to set the seed to. Naturally, coders should always choose the number 42 ;)

    """
    random.seed(seed)
    np.random.seed(seed)

def generate_binary_cluster_X_and_y(m, n, seed):
    """
    This function generates the design matrix X and target vector y that will be used for training the model.
    Do not alter this function! The code to generate the dataset you will use is already set up.

    Parameters
    ----------
    m : int
        The number of datapoints we will generate for each cluster of datapoints.
    n : int
        The number of feature columns we will generate.
    seed : int
        The value to set the seed to. Naturally, coders should always choose the number 42 ;)

    """
    # Set the random seed
    Set_Seed(seed)
    #Generate a mean vector for both clusters
    mean_0 = np.random.normal([1 for i in range(n)])
    mean_1 = np.random.normal([5 for i in range(n)])
    # Generate a random mxn design matrix X
    X_0 = 100*np.random.normal(loc=mean_0, size=(m,n))
    X_1 = 100*np.random.normal(loc=mean_1, size=(m,n))
    #Generate the target vector
    y_0 = [0 for i in range(m)]
    y_1 = [1 for i in range(m)]
    #Append the X and y arrays
    X = np.vstack((X_0, X_1))
    y = np.asarray(y_0 + y_1)
    #Now, append a column of ones to be the first column of X; this will correpsond to the intercept
    ones_column = np.ones((X.shape[0], 1))
    X = np.hstack((ones_column, X))
    return X, y


###############################################################################
############################## Plotting Functions #############################
###############################################################################

def plot_3d_scatter_with_hyperplane(learned_theta, X, y):
    #First, generate a sklearn linear model as the true model we are trying to learn
    log_mod = LogisticRegression().fit(pd.DataFrame(X[:,1:]), pd.DataFrame(y)) 
    sk_theta = log_mod.coef_
    intercept = log_mod.intercept_
    sk_theta = np.asarray([intercept[0], sk_theta[0][0], sk_theta[0][1]])
    # Assuming X1, X2 are your features and y is the target
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    # Scatter the original data points
    #sns.scatterplot(x=X[:, 1], y=X[:, 2], hue=y, label='Data points')
    ax.scatter(X[y == 0][:, 1], X[y == 0][:, 2], c='red', label='Class 0')
    ax.scatter(X[y == 1][:, 1], X[y == 1][:, 2], c='blue', label='Class 1')
    # Create a meshgrid for X1 and X2 to plot the plane
    X1_mesh, X2_mesh = np.meshgrid(np.linspace(min(X[:, 1]), max(X[:, 1]), 1000), np.linspace(min(X[:, 2]), max(X[:, 2]), 1000))
    if len(learned_theta) != 0:
        # Plot the decision boundary (theta_0 + theta_1 * x1 + theta_2 * x2 = 0)
        # Rearranged as x2 = -(theta_0 + theta_1 * x1) / theta_2
        sk_x1_values = np.array([min(X[:, 1]), max(X[:, 1])])
        sk_x2_values = -(sk_theta[0] + sk_theta[1] * sk_x1_values) / sk_theta[2]
        # Plot the optimal regression plane
        ax.plot(sk_x2_values, sk_x1_values, label='Optimal logistic Regression Decision Boundary', color='green')
        # learned decision boundary
        learned_x1_values = np.array([min(X[:, 1]), max(X[:, 1])])
        learned_x2_values = -(learned_theta[0] + learned_theta[1] * sk_x1_values) / learned_theta[2]
        # Plot the optimal regression plane
        ax.plot(learned_x1_values, learned_x2_values, label='Learned logistic Regression Decision Boundary', color='purple')
        #Axis and labels
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title('Logistic Regression Decision Boundary')
        plt.legend()
        plt.show()
    else:
        # Plot the decision boundary (theta_0 + theta_1 * x1 + theta_2 * x2 = 0)
        # Rearranged as x2 = -(theta_0 + theta_1 * x1) / theta_2
        sk_x1_values = np.array([min(X[:, 1]), max(X[:, 1])])
        sk_x2_values = -(sk_theta[0] + sk_theta[1] * sk_x1_values) / sk_theta[2]
        # Plot the optimal regression plane
        ax.plot(sk_x2_values, sk_x1_values, label='Optimal logistic Regression Decision Boundary', color='green')
        #Axis and labels
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title('Logistic Regression Decision Boundary')
        plt.legend()
        plt.show()


###############################################################################
########################### Student Written Functions #########################
###############################################################################

# def sigmoid(z):
    """
    This function computes the vector of sigmoid values given the input dot-product z = theta^T x.

    Parameters
    ----------
    z : np.array
        The affinity vector defined by z = theta^T x.

    """
    ######################################
    ########### YOUR CODE HERE ###########
    ######################################

# def cross_entropy_loss_logistic(theta, X, y, epsilon=1e-12):
    """
    This function computes the cross-entropy loss function for a logistic model.

    Parameters
    ----------
    theta : np.array
        The parameter vector.
    X : np.array
        The design matrix that we generated.
    y : np.array
        The target vector that we generated.
    epsilon : float
        This is the value used to clip very small log values of log(0) to.

    """
    # # Compute the logistic regression model's predictions
    
    ######################################
    ########### YOUR CODE HERE ###########
    ######################################
    
    # # Clip predictions to prevent log(0)
    # preds = np.clip(preds, epsilon, 1 - epsilon)
    
    #Compute the cross-entropy loss
    
    ######################################
    ########### YOUR CODE HERE ###########
    ######################################


# def grad_ce_loss_logistic(theta, X, y):
    """
    This function computes the gradient vector of the cross-entropy loss function for a logistic model.

    Parameters
    ----------
    theta : np.array
        The parameter vector.
    X : np.array
        The design matrix that we generated.
    y : np.array
        The target vector that we generated.

    """
    ######################################
    ########### YOUR CODE HERE ###########
    ######################################


# def opt_algorithm(X, y, init_theta, init_alpha, max_iter, algo, alpha_scheme):
    """
    This function performs gradient descent (or Newton's method) to minimize
    the cross-entropy loss function for a logistic model to obtain an optimal
    parameter vector instead of using the normal equations.

    Parameters
    ----------
    X : np.array
        The design matrix that we generated.
    y : np.array
        The target vector that we generated.
    init_theta : np.array
        The initial parameter vector to be optimized.
    init_alpha : float
        The initial learning rate. This should be a value < 1.
    max_iter : int
        The maximum number of iterations to train the model for (the number of time you update the parameters).
    algo : string
        The algorithm to be used. Valid input sould be 'steepest' (or also 'Newton' for graduate students)
    alpha_scheme : string
        The scheme used to choose the learning rate alpha each iteration.
        Valid choices are 'fixed', 'decay', or 'log_decay'.

    """
#    theta = init_theta
#    alpha = init_alpha
#    iter_list = []
#    loss_list = []
#    ######################################
#    ########### YOUR CODE HERE ###########
#    ######################################
#    return theta, iter_list, loss_list


###############################################################################
########################### Numerical Results Section #########################
###############################################################################

########## DO NOT CHANGE THIS BLOCK OF CODE (it generates the data you will use as well as an initial value of theta)
Set_Seed(42)
m = 500 #Number of datapoints
n = 2 #Number of features
X, y = generate_binary_cluster_X_and_y(m, n, 42) #Generate a valid design matrix
init_theta = 2*np.random.randn(n+1)
##########


###############################################################################
################ Your code below here (uncomment where needed) ################
# init_alpha = 
# max_iter = 
# algo = 'steepest'
# alpha_scheme = 
# learned_theta, iter_list, loss_list = opt_algorithm(X, y, init_theta, init_alpha, max_iter, algo, alpha_scheme)


# # Plot the performance of the model over the training iterations
# sns.lineplot(x=iter_list, y=loss_list, color='blue')
# ax = plt.gca()
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Cross Entropy Loss')
# ax.set_title('Logistic Regression Training')
# # Manually add legend for the regression plane
# gradient_loss_proxy = plt.Line2D([0], [0], linestyle="-", c='blue', marker='none')
# ax.legend([gradient_loss_proxy], ['Gradient Descent Training Loss'])
# plt.show()




###############################################################################
############################ Visualization Section ############################
###############################################################################

# (Again, uncomment where needed)

# Run this if you haven't have trained a logistic regressin model via gradient descent yet
plot_3d_scatter_with_hyperplane(np.array([]), X, y)

# # Run this if you have trained a logistic regression model via gradient descent with the parameters stored in the np.array learned_weights
# plot_3d_scatter_with_hyperplane(learned_theta, X, y)





