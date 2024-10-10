#ISE 364/464 Homework 3 - Problem 4
#Implementation of training a Linear Regression Model via gradient descent
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


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

def generate_correlated_PD_X_and_y(m, n, corr_strength, seed):
    """
    This function generates the design matrix X and target vector y that will be used for training the model.
    Do not alter this function! The code to generate the dataset you will use is already set up.
    This function generates the data in such a way such that X will have full-column rank.

    Parameters
    ----------
    m : int
        The number of datapoints we will generate.
    n : int
        The number of feature columns we will generate.
    corr_strength : float
        This is a value in the range [-1,1] and will dictate how strong of a linear relationship will exist between the features and the target.
    seed : int
        The value to set the seed to. Naturally, coders should always choose the number 42 ;)

    """
    # Set the random seed
    Set_Seed(seed)
    # Create a covariance matrix for the multivariate normal distribution; the covariance matrix must be positive semi-definite
    cov_matrix = (1 - corr_strength) * np.eye(n) + corr_strength * np.ones((n, n))
    # Generate a random mxn design matrix X with correlated columns
    X = 100*np.random.multivariate_normal(mean=np.zeros(n), cov=cov_matrix, size=m)
    # Generate a random weight vector to mix columns of X to form y
    weights = np.random.randn(n)
    # Generate z as a linear combination of the columns of X with some added noise
    noise = 350*np.random.randn(m) * (1 - corr_strength)
    y = X @ weights + noise
    #Verify that the design matrix is valid
    XtX=np.dot(X.T, X)
    eigenvals = np.linalg.eigvals(XtX)
    if np.all(eigenvals > 0):
        print("Generated a design matrix X such that X^T X is PD.")
    else:
        print("Failed to generate a design matrix X such that X^T X is PD. Run again with a different seed")
    #Now, append a column of ones to be the first column of X; this will correpsond to the intercept
    ones_column = np.ones((X.shape[0], 1))
    X = np.hstack((ones_column, X))
    return X, y


###############################################################################
############################## Plotting Functions #############################
###############################################################################

def plot_lin_reg_contours(X, y):
    """
    This function generates a contour plot of the least-squares fit line to the datapoints (which are colored by the true value of y).
    This is one way of visualizing a hyperplane.
    As it is set up, it will make the least squares linear model using sklearn for convenience. This is simply meant to help give you visual aid.

    Parameters
    ----------
    X : np.array
        The design matrix that we generated.
    y : np.array
        The target vector that we generated.

    """
    #First, generate a sklearn linear model as the true model we are trying to learn
    #We could use the normal equations, but this way it is not dependent on the students to define the normal equations correctly
    lin_mod = LinearRegression().fit(pd.DataFrame(X[:,1:]), pd.DataFrame(y)) 
    sk_theta = lin_mod.coef_
    intercept = lin_mod.intercept_
    sk_theta = np.asarray([intercept[0], sk_theta[0][0], sk_theta[0][1]])
    # Generate a meshgrid for the features X1 and X2
    X1_range = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    X2_range = np.linspace(min(X[:, 2]), max(X[:, 2]), 100)
    X1_mesh, X2_mesh = np.meshgrid(X1_range, X2_range)
    # Predict the values on the meshgrid
    y_pred_mesh = sk_theta[0] + sk_theta[1] * X1_mesh + sk_theta[2] * X2_mesh
    # Create the contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X1_mesh, X2_mesh, y_pred_mesh, cmap='magma', alpha=0.75)
    plt.colorbar(contour, label='Predicted $y$')
    # Scatter plot of the original data points
    plt.scatter(X[:, 1], X[:, 2], c=y, edgecolor='k', cmap='coolwarm', label='Data Points (Colored by True $y$)')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Contour Plot of Linear Regression Predictions')
    plt.legend()
    plt.show()

def plot_3d_scatter_with_hyperplane(learned_theta, X, y):
    #First, generate a sklearn linear model as the true model we are trying to learn
    #We could use the normal equations, but this way it is not dependent on the students to define the normal equations correctly
    lin_mod = LinearRegression().fit(pd.DataFrame(X[:,1:]), pd.DataFrame(y)) 
    sk_theta = lin_mod.coef_
    intercept = lin_mod.intercept_
    sk_theta = np.asarray([intercept[0], sk_theta[0][0], sk_theta[0][1]])
    # Assuming X1, X2 are your features and y is the target
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    # Scatter the original data points
    ax.scatter(X[:, 1], X[:, 2], y, color='blue', label='Data points')
    # Create a meshgrid for X1 and X2 to plot the plane
    X1_mesh, X2_mesh = np.meshgrid(np.linspace(min(X[:, 1]), max(X[:, 1]), 500), np.linspace(min(X[:, 2]), max(X[:, 2]), 500))
    sklearn_y_pred_mesh = sk_theta[0] + sk_theta[1] * X1_mesh + sk_theta[2] * X2_mesh
    if len(learned_theta) != 0:
        learned_y_pred_mesh = learned_theta[0] + learned_theta[1] * X1_mesh + learned_theta[2] * X2_mesh
        # Plot the true regression plane
        ax.plot_surface(X1_mesh, X2_mesh, sklearn_y_pred_mesh, color='red', alpha=0.5, label='Optimal Linear Regression Model (Normal Equations)')
        # Plot the learned regression plane
        ax.plot_surface(X1_mesh, X2_mesh, learned_y_pred_mesh, color='green', alpha=0.5, label='Learned Linear Regression Model (Gradient Descent)')
        #Axis and labels
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$y$')
        ax.set_title('Linear Regression Model')
        # Manually add legend for the regression plane
        scatter_proxy = plt.Line2D([0], [0], linestyle="none", c='blue', marker='o')
        sk_plane_proxy = plt.Line2D([0], [0], linestyle="none", c='red', marker='s')
        learned_plane_proxy = plt.Line2D([0], [0], linestyle="none", c='green', marker='s')
        ax.legend([scatter_proxy, sk_plane_proxy, learned_plane_proxy], ['Data Points', 'Optimal Linear Regression Model (Normal Equations)', 'Learned Linear Regression Model (Gradient Descent)'])
        plt.show()
    else:
        ax.plot_surface(X1_mesh, X2_mesh, sklearn_y_pred_mesh, color='red', alpha=0.5, label='Optimal Linear Regression Model (Normal Equations)')
        #Axis and labels
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$y$')
        ax.set_title('Linear Regression Model')
        # Manually add legend for the regression plane
        scatter_proxy = plt.Line2D([0], [0], linestyle="none", c='blue', marker='o')
        sk_plane_proxy = plt.Line2D([0], [0], linestyle="none", c='red', marker='s')
        ax.legend([scatter_proxy, sk_plane_proxy], ['Data Points', 'Optimal Linear Regression Model (Normal Equations)'])
        plt.show()


###############################################################################
########################### Student Written Functions #########################
###############################################################################

# def mse_loss_linear(theta, X, y):
    """
    This function computes the MSE loss function for a linear model.

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


# def grad_mse_linear(theta, X, y):
    """
    This function computes the gradient vector of the MSE loss function for a linear model.

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


# def optimal_lin_reg_sol(X, y):
    """
    This function computes the optimal solution vector of parameters for a least-squares linear regression model via the normal equations.

    Parameters
    ----------
    X : np.array
        The design matrix that we generated.
    y : np.array
        The target vector that we generated.

    """
#     ######################################
#     ########### YOUR CODE HERE ###########
#     ######################################
#     return theta_star


# def gradient_desc(X, y, init_theta, init_alpha, max_iter):
    """
    This function performs gradient descent to minimize the MSE loss function for a linear model to obtain an optimal parameter vector instead of using the normal equations.

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

    """
#     theta = init_theta
#     alpha = init_alpha
#     iter_list = []
#     loss_list = []
#     ######################################
#     ########### YOUR CODE HERE ###########
#     ######################################
#     return final_theta, iter_list, loss_list


###############################################################################
########################### Numerical Results Section #########################
###############################################################################

########## DO NOT CHANGE THIS BLOCK OF CODE (it generates the data you will use as well as an initial value of theta)
Set_Seed(42)
m = 500 #Number of datapoints
n = 2 #Number of features
corr_strength = 0.8 #Strength of the linear relationships
X, y = generate_correlated_PD_X_and_y(m, n, corr_strength, 42) #Generate a valid design matrix
init_theta = 10*np.random.randn(n+1)
##########


###############################################################################
################ Your code below here (uncomment where needed) ################
# init_alpha = 
# max_iter = 
# learned_theta, iter_list, loss_list = gradient_desc(X, y, init_theta, init_alpha, max_iter)


# # Optimal solution theta*
# theta_star = optimal_lin_reg_sol(X, y)
# opt_loss = mse_loss_linear(theta_star, X, y)


# # Plot the performance of the model over the training iterations
# sns.lineplot(x=iter_list, y=loss_list, color='blue')
# ax = plt.gca()
# ax.axhline(y=opt_loss, color='red', linestyle='--', label='Optimal MSE Loss')
# ax.set_xlabel('Iterations')
# ax.set_ylabel('MSE Loss')
# ax.set_title('Linear Regression Training')
# # Manually add legend for the regression plane
# gradient_loss_proxy = plt.Line2D([0], [0], linestyle="-", c='blue', marker='none')
# optimal_proxy = plt.Line2D([0], [0], linestyle="--", c='red', marker='none')
# ax.legend([gradient_loss_proxy, optimal_proxy], ['Gradient Descent Training Loss', 'Optimal MSE Loss'])
# plt.show()




###############################################################################
############################ Visualization Section ############################
###############################################################################

# (Again, uncomment where needed)

# Plots the 2D scatterplot of the dataset (colored by the true value of y) along with the contours of the optimal linear regression model
plot_lin_reg_contours(X, y)

# Run this if you haven't have trained a linear regressin model via gradient descent yet
plot_3d_scatter_with_hyperplane(np.array([]), X, y)

# # Run this if you have trained a linear regression model via gradient descent with the parameters stored in the np.array learned_weights
# plot_3d_scatter_with_hyperplane(learned_theta, X, y)
