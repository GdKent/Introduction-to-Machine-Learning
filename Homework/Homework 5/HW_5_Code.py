#ISE 364/464 Homework 5 - Problem 4
#Implementation of Decision Trees, Random Forests, and Gradient Boosting
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


###############################################################################
###################$ Dataset Generation & Plotting Functions ##################
###############################################################################

def Set_Seed(seed):
    """
    Simple function to set the random seed for the relevant libraries that one will be using.

    Parameters
    ----------
    seed : int
        The value to set the seed to. Naturally, coders should always choose the number 42 ;)

    """
    random.seed(seed)
    np.random.seed(seed)

def Generate_Dataset(n_datapoints=500, centers=3, cluster_std=5, seed=42):
    """
    Function to generate a two-dimensional dataset of overlapping clusters.

    Parameters
    ----------
    n_datapoints : int
        The number of datapoints to generate.
    centers : int
        The number of clusters (the number of target classes).
    cluster_std : float
        The standard deviation of each cluster. This value should be large enough to ensure that there is class overlap.
    seed : int
        The value to set the seed to. Naturally, coders should always choose the number 42 ;)

    """
    Set_Seed(seed)
    #Generate the 2-D dataset of overlapping clusters
    X, y = make_blobs(n_samples=n_datapoints, centers=centers, random_state=seed, cluster_std=cluster_std)
    #Generate a scatter plot to visualize the resulting data
    ax = plt.gca()
    ax.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=20, alpha=0.8, edgecolors='black', cmap='magma')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Overlapping Clusters Dataset')
    return X, y

def Plot_Train_Val_w_Decision(X_train, y_train, X_val, y_val, model, model_name='(model name here)'):
    """
    Function to plot a the training and validation datasets as well as the decision boundary of a given classification model.

    Parameters
    ----------
    X_train : np.array
        The design matrix of training data.
    y_train : np.array
        The target vector of training data.
    X_val : np.array
        The design matrix of validation data.
    y_val : np.array
        The target vector of validation data.
    model : sklearn model
        A trained sklearn model that will be used to generate predictions.
    model_name : string
        The name of the classification model being used (this is simply to include the name of the model on the plot).

    """
    if model != None:
        #Compute the model's accuracy on both the training and validation datasets
        y_pred_train = model.predict(X_train); y_pred_val = model.predict(X_val)
        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)
        #Scatterplot of the training and validation datasets, side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=False)
        axes[0].scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train, s=20, alpha=0.8, edgecolors='black', cmap='magma')
        xlim = axes[0].get_xlim()
        ylim = axes[0].get_ylim()
        # Fit the estimator
        xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        # Create a color plot with the results
        n_classes = len(np.unique(y))
        axes[0].contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes + 1) - 0.5, cmap='magma', clim=(y.min(), y.max()), zorder=1)
        axes[0].set(xlim=xlim, ylim=ylim)
        axes[0].set_xlabel("$x_1$")
        axes[0].set_ylabel("$x_2$")
        axes[0].set_title(f'{model_name} Decision Boundary \n (Training Dataset) Total Accuracy: {train_acc}')
        
        axes[1].scatter(X_valid[:, 0], X_valid[:, 1], marker='o', c=y_valid, s=20, alpha=0.8, edgecolors='black', cmap='magma')
        xlim = axes[1].get_xlim()
        ylim = axes[1].get_ylim()
        # Fit the estimator
        xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        # Create a color plot with the results
        n_classes = len(np.unique(y))
        axes[1].contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes + 1) - 0.5, cmap='magma', clim=(y.min(), y.max()), zorder=1)
        axes[1].set(xlim=xlim, ylim=ylim)
        axes[1].set_xlabel("$x_1$")
        axes[1].set_ylabel("$x_2$")
        axes[1].set_title(f'{model_name} Decision Boundary \n (Validation Dataset) Total Accuracy: {val_acc}')
        plt.tight_layout()
    else:
        #Scatterplot of the training and validation datasets, side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=False)
        axes[0].scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train, s=20, alpha=0.8, edgecolors='black', cmap='magma')
        axes[0].set_xlabel("$x_1$")
        axes[0].set_ylabel("$x_2$")
        axes[0].set_title("Training Dataset")
        
        axes[1].scatter(X_valid[:, 0], X_valid[:, 1], marker='o', c=y_valid, s=20, alpha=0.8, edgecolors='black', cmap='magma')
        axes[1].set_xlabel("$x_1$")
        axes[1].set_ylabel("$x_2$")
        axes[1].set_title("Validation Dataset")
        plt.tight_layout()


###############################################################################
############################ Data Generation Section ##########################
###############################################################################

########## DO NOT CHANGE THIS BLOCK OF CODE (it generates the data you will use as well as an initial value of theta)
Set_Seed(42)
# Make the dataset
X, y = Generate_Dataset(n_datapoints=500, centers=3, cluster_std=5)
# Generate a simple 80%-20% train-test split of the data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
print('Shape of training set: '+str(X_train.shape))
print('Shape of validation set: '+str(X_valid.shape))
#Scatterplot of the training and validation datasets, side-by-side
Plot_Train_Val_w_Decision(X_train, y_train, X_valid, y_valid, model=None)
##########




###############################################################################
######################## Machine Learning Classification ######################
###############################################################################

########################### Decision Tree Classifier ##########################
################ Your code below here (uncomment where needed) ################
# Set_Seed(42)
# from sklearn.tree import DecisionTreeClassifier
# ######################################
# ########### YOUR CODE HERE ###########
# ######################################
# Plot_Train_Val_w_Decision(X_train, y_train, X_valid, y_valid, model= , model_name='Decision Tree')




########################### Random Forest Classifier ##########################
################ Your code below here (uncomment where needed) ################
# Set_Seed(42)
# from sklearn.ensemble import RandomForestClassifier
# ######################################
# ########### YOUR CODE HERE ###########
# ######################################
# Plot_Train_Val_w_Decision(X_train, y_train, X_valid, y_valid, model= , model_name='Random Forest')




####################### Gradient Boosted Trees Classifier #####################
################ Your code below here (uncomment where needed) ################
# Set_Seed(42)
# from sklearn.ensemble import GradientBoostingClassifier
# ######################################
# ########### YOUR CODE HERE ###########
# ######################################
# Plot_Train_Val_w_Decision(X_train, y_train, X_valid, y_valid, model= , model_name='Gradient Boosted Trees')

