"""
Implementations of all the needed methods to run
the project, including the 6 asked methods (The first 6).
"""
import numpy as np
from helpers import batch_iter

##############################################################################################################
###############################  Mandatory Functions     #####################################################
##############################################################################################################

"""
Linear regression using gradient descent
"""
def least_squares_GD(y, tx, initial_w, max_iters, gamma, ltype="MSE"):
    # ***************************************************
    ws = initial_w  # initiate weights
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, ws)  # compute the gradient for current weights
        # update w by gradient
        ws = ws - gamma * gradient  # updates the new weights

    loss = compute_loss(y, tx, ws, ltype)  # compute final error

    return ws, loss
    # ***************************************************

"""
Linear regression using stochastic gradient descent
"""
def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    # ***************************************************
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=1):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            # update w by gradient
            w = w - gamma * gradient  # computes the new w(t+1)

    loss = compute_loss(y, tx, w)
    return w, loss
    # ***************************************************

"""
Least squares regression using normal equations
"""
def least_squares(y, tx):
    # ***************************************************
    w = np.linalg.solve((tx.transpose()).dot(tx), (tx.transpose()).dot(y))
    loss = compute_loss(y, tx, w)

    return w, loss
    # ***************************************************

"""
Ridge regression using normal equations
"""
def ridge_regression(y, tx, lambda_):
    # ***************************************************
    tx_T = tx.transpose()
    a = np.dot(tx_T,tx) + 2*len(y)*lambda_*np.identity(tx.shape[1])
    b = np.dot(tx_T,y)
    ws = np.linalg.solve(a,b)
    loss = compute_loss(y, tx, ws)

    return ws, loss
    # ***************************************************

"""
Logistic regression using gradient descent
"""
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # ***************************************************
    w = initial_w

    for i in range(max_iters):
        grad = np.zeros(tx.shape[1])

        sigma = sigmoid(tx.dot(w))
        SX = tx * (sigma - sigma*sigma).reshape(-1,1)
        XSX = tx.transpose().dot(SX)

        for aw in range(tx.shape[0]):
            grad = grad + (-1 / tx.shape[0]) * (y[aw] * tx[aw,:] * sigmoid(-y[aw] * np.dot(tx[aw,:], w)))

        w = w - gamma * np.linalg.solve(XSX, grad)

        if i % 5 == 0 and i != 0:
            gamma = gamma * 0.55

    loss = compute_logistic_loss(y, tx, w)

    return w, loss
    # ***************************************************

"""
Regularized logistic regression using gradient descent
"""
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # ***************************************************
    w = initial_w

    for i in range(max_iters):
        grad = np.zeros(tx.shape[1])

        sigma = sigmoid(tx.dot(w))
        SX = tx * (sigma - sigma*sigma).reshape(-1,1)
        XSX = tx.transpose().dot(SX) + lambda_*np.eye((tx.shape[1]))

        for aw in range(tx.shape[0]):
            grad = grad + (-1 / tx.shape[0]) * (y[aw] * tx[aw,:] * sigmoid(-y[aw] * np.dot(tx[aw,:],w)))

        w = w - gamma * np.linalg.solve(XSX, grad) - gamma * lambda_*w

        if i % 5 == 0 and i != 0:
            gamma = gamma * 0.55

    loss = compute_logistic_loss(y, tx, w)

    return w, loss
    # ***************************************************

"""
Computes the MSE of the given weights applied to tx
"""
def compute_loss(y, tx, w):
    # ***************************************************
    e = y - np.dot(tx, w)
    mse = np.dot(e.transpose(), e) / (2 * len(tx))

    return mse
    # ***************************************************

"""
Computes the Logistic loss for given w by MSE using the sigmoid
"""
def compute_logistic_loss(y, tx, w, ltype="MSE"):
    """Calculate the loss.You can calculate the loss using mse or mae"""
    pred = sigmoid(tx.dot(w))
    if (ltype=="MAE"):
    # ***************************************************
    # Lost function by MAE
        e = y - pred
        loss = np.mean(np.abs(e))
    else:
    # ***************************************************
    # Lost function by MSE
        loss = (((y - pred)**2).mean(axis = 0)) / 2 #Loss calculated using MSE
    return loss

"""
Sigmoid function for a scalar - This enables us to prevent
from overflow in case the function is growing too strongly  
"""
def sig(a):
    # ***************************************************
    if a > 0:
        return 1.0 / (1 + np.exp(-a))
    else:
        return np.exp(a) / (1.0 + np.exp(a))
    # ***************************************************
"""
The vectorized version of sigmoid
"""
sigmoid = np.vectorize(sig)

"""
Computes the gradient with respect to y, tx and w
"""
def compute_gradient(y, tx, w):
    # ***************************************************
    N = len(y)
    gradient = -((1 / N) * (np.dot(np.transpose(tx), y - np.dot(tx, w))))

    return gradient
    # ***************************************************

##############################################################################################################
############### GD and SGD using MAE  ########################################################################
##############################################################################################################

def compute_subgradient(y, tx, w):
    # ***************************************************
    # compute gradient and loss
    N = len(y)

    # MAE gradient:
    gradient = -((1 / N) * (np.dot(np.transpose(tx), np.sign(y - np.dot(tx, w)))))
    return gradient


def compute_loss_subgradient(y, tx, w):
    """Calculate the loss using MAE"""
    # ***************************************************
    # Lost function by MAE
    losses = (np.abs((y - np.dot(tx, w)))).mean(axis=0)
    return losses


def subgradient_descent(y, tx, initial_w, max_iters, gamma, ltype="MSE"):
    """Gradient descent algorithm."""
    ws = initial_w  # initiate weights
    for n_iter in range(max_iters):
        gradient = compute_subgradient(y, tx, ws)  # compute the gradient for current weights
        # update w by gradient
        ws = ws - gamma * gradient  # updates the new weights

    loss = compute_loss_subgradient(y, tx, ws)  # compute final error

    return ws, loss


def stochastic_subgradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, ltype="MAE"):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # implement stochastic gradient descent.
    w = initial_w
    g = 0
    num_batches = 1
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches):
            g = compute_subgradient(minibatch_y, minibatch_tx, w)
            # update w by gradient
            w = w - gamma * g  # computes the new w(t+1)
    loss = compute_loss_subgradient(y, tx, w)  # compute final error

    return w, loss


##############################################################################################################
###############################  Standardization Methods     #################################################
##############################################################################################################

def standardize(centered_tX):
    centered_tX[centered_tX==0] = float('nan')
    stdevtrain = np.nanstd(centered_tX, axis=0)
    centered_tX[centered_tX==float('nan')] = 0
    stdevtrain[stdevtrain == 0] = 0.00001
            #CHECK WHY IT IS HAPPENING
    standardized_tX = centered_tX / stdevtrain
    return standardized_tX, stdevtrain

def standardize_original(tX):
    # Removing bothering data and centering
    tX[tX==-999] = 0
    s_mean = np.mean(tX, axis=0)
    centered_tX = tX - s_mean
    stdtX, stdevtrain = standardize(centered_tX)

    return stdtX, stdevtrain, s_mean

def standardize_basis(tX):
    # Resetting all the data
    b_mean = np.mean(tX,axis=0)
    centered_mat = tX - b_mean
    centered_mat[tX==0] = 0
    standardized_tX, stdevtrain = standardize(centered_mat)

    return standardized_tX, stdevtrain, b_mean

def standardize_test_original(tX, training_original_mean, stdevtrain):
    tX[tX==-999] = 0
    centered_testx = tX - training_original_mean
    centered_testx[tX==-999] = 0
    standardized_testx = centered_testx / stdevtrain

    return standardized_testx

def standardized_testx_basis(tX, basis_original_mean, stdev):
    centered_mat = tX - basis_original_mean
    centered_mat[tX==0] = 0
    standardized_testmat = centered_mat / stdev

    return standardized_testmat

"""
Returns a polynomial basis formed of all the degrees and combinations.
- The first part of the code computes from degree 1 to a given degree (max d=15)
- The second part of the code, computes the second degree with combinations meanning
that is combines every feature with each other in order to make paramters like the PHI angles
more meaningful.
- Finally, The last one is the third degree basis with combinations of elements
that do not all have the same degree, and only taking the 15 first most meaningful
features for our model. 
"""
def build_poly_basis(tx):
    d = len(tx[0])
    n = len(tx)

    indices_s_deg = []
    indices_t_deg = []

    print("Creating indices for subsets of degree 2")
    for i in range (d):
        for t in range (i,d):
            indices_s_deg.append([t, i])
    indices_s_deg = np.array(indices_s_deg).T

    print("Creating indices for subsets of degree 3")
    max_t_degree = min(d-1,15)
    for i in range (max_t_degree):
        for t in range (i,max_t_degree):
            for j in range(t,max_t_degree):
                if not (i == t and i == j):
                    indices_t_deg.append([j, t, i])
    indices_t_deg = np.array(indices_t_deg).T

    degrees = range(3,11)
    degrees_number = len(degrees) + 1
    stdX_Ncols = tx.shape[1]
    indices_s_Ncols = indices_s_deg.shape[1]
    indices_t_Ncols = indices_t_deg.shape[1]

    number_of_rows = indices_s_Ncols + degrees_number * stdX_Ncols + indices_t_Ncols

    mat = np.zeros((n, number_of_rows))

    print("Computing first degree")
    # First degree
    mat[:, :stdX_Ncols] = tx

    print("Computing second degree WITH combinations")
    # Second degree gotten from indices
    mat[:,stdX_Ncols:stdX_Ncols + indices_s_Ncols] = tx[:, indices_s_deg[0]] * tx[:, indices_s_deg[1]]

    print("Computing from degree 3 to 10 WITHOUT combinations...")
    # Improve 3 to 10 degree
    for i in degrees:
        start_index = indices_s_Ncols + (i - 2) * stdX_Ncols
        end_index = start_index + stdX_Ncols
        mat[:,start_index:end_index] = tx**i

    print("Computing third degree WITH combinations...")
    # Third degree gotten from indices
    mat[:, number_of_rows - indices_t_Ncols: number_of_rows] = tx[:, indices_t_deg[0]] * tx[:, indices_t_deg[1]] * tx[:, indices_t_deg[2]]

    return mat


##############################################################################################################
###############################  Additional Functions    #####################################################
##############################################################################################################

def cross_validation_prep_zeros(y, tx):
    x1 = np.delete(tx, range(80000, 99913), axis=0)
    x2 = np.delete(tx, range(60000, 80000), axis=0)
    x3 = np.delete(tx, range(40000, 60000), axis=0)
    x4 = np.delete(tx, range(20000, 40000), axis=0)
    x5 = np.delete(tx, range(0, 20000), axis=0)
    y1 = np.delete(y, range(80000, 99913), axis=0)
    y2 = np.delete(y, range(60000, 80000), axis=0)
    y3 = np.delete(y, range(40000, 60000), axis=0)
    y4 = np.delete(y, range(20000, 40000), axis=0)
    y5 = np.delete(y, range(0, 20000), axis=0)

    return x1, x2, x3, x4, x5, y1, y2, y3, y4, y5


def cross_validation_prep_ones(y, tx):
    x1 = np.delete(tx, range(50000, 77544), axis=0)
    x2 = np.delete(tx, range(35000, 50000), axis=0)
    x3 = np.delete(tx, range(20000, 35000), axis=0)
    x4 = np.delete(tx, range(5000, 20000), axis=0)
    x5 = np.delete(tx, range(0, 15000), axis=0)
    y1 = np.delete(y, range(50000, 77544), axis=0)
    y2 = np.delete(y, range(35000, 50000), axis=0)
    y3 = np.delete(y, range(20000, 35000), axis=0)
    y4 = np.delete(y, range(5000, 20000), axis=0)
    y5 = np.delete(y, range(0, 15000), axis=0)

    return x1, x2, x3, x4, x5, y1, y2, y3, y4, y5


def cross_validation_prep_twos(y, tx):
    x1 = np.delete(tx, range(50000, 72543), axis=0)
    x2 = np.delete(tx, range(35000, 50000), axis=0)
    x3 = np.delete(tx, range(20000, 35000), axis=0)
    x4 = np.delete(tx, range(5000, 20000), axis=0)
    x5 = np.delete(tx, range(0, 15000), axis=0)
    y1 = np.delete(y, range(50000, 72543), axis=0)
    y2 = np.delete(y, range(35000, 50000), axis=0)
    y3 = np.delete(y, range(20000, 35000), axis=0)
    y4 = np.delete(y, range(5000, 20000), axis=0)
    y5 = np.delete(y, range(0, 15000), axis=0)

    return x1, x2, x3, x4, x5, y1, y2, y3, y4, y5


def cross_validation_prep_zeros_small(y, tx):
    x1 = np.delete(tx, range(3000, 4016), axis=0)
    x2 = np.delete(tx, range(2000, 3000), axis=0)
    x3 = np.delete(tx, range(1000, 2000), axis=0)
    x4 = np.delete(tx, range(0, 1000), axis=0)
    x5 = np.delete(tx, range(500, 1500), axis=0)
    y1 = np.delete(y, range(3000, 4016), axis=0)
    y2 = np.delete(y, range(2000, 3000), axis=0)
    y3 = np.delete(y, range(1000, 2000), axis=0)
    y4 = np.delete(y, range(0, 1000), axis=0)
    y5 = np.delete(y, range(500, 1500), axis=0)

    return x1, x2, x3, x4, x5, y1, y2, y3, y4, y5


def cross_validation_prep_ones_small(y, tx):
    x1 = np.delete(tx, range(2500, 3045), axis=0)
    x2 = np.delete(tx, range(2000, 2500), axis=0)
    x3 = np.delete(tx, range(1500, 2000), axis=0)
    x4 = np.delete(tx, range(1000, 1500), axis=0)
    x5 = np.delete(tx, range(0, 1000), axis=0)
    y1 = np.delete(y, range(2500, 3045), axis=0)
    y2 = np.delete(y, range(2000, 2500), axis=0)
    y3 = np.delete(y, range(1500, 2000), axis=0)
    y4 = np.delete(y, range(1000, 1500), axis=0)
    y5 = np.delete(y, range(0, 1000), axis=0)

    return x1, x2, x3, x4, x5, y1, y2, y3, y4, y5


def cross_validation_prep_twos_small(y, tx):
    x1 = np.delete(tx, range(2500, 2938), axis=0)
    x2 = np.delete(tx, range(2000, 2500), axis=0)
    x3 = np.delete(tx, range(1500, 2000), axis=0)
    x4 = np.delete(tx, range(1000, 1500), axis=0)
    x5 = np.delete(tx, range(0, 1000), axis=0)
    y1 = np.delete(y, range(2500, 2938), axis=0)
    y2 = np.delete(y, range(2000, 2500), axis=0)
    y3 = np.delete(y, range(1500, 2000), axis=0)
    y4 = np.delete(y, range(1000, 1500), axis=0)
    y5 = np.delete(y, range(0, 1000), axis=0)

    return x1, x2, x3, x4, x5, y1, y2, y3, y4, y5


def sliptJets(y, tx, tx_std, ids):
    x_jet_zeros = []
    x_jet_ones = []
    x_jet_twos = []

    y_jet_zeros = []
    y_jet_ones = []
    y_jet_twos = []

    ids_jet_zeros = []
    ids_jet_ones = []
    ids_jet_twos = []

    print('hello', tx.shape)
    print (tx_std.shape)
    shape = tx_std.shape

    print ('shape', shape)

    for i in range(shape[0]):  # LOOPS CAN CERTAINLY BE OPTIMIZED
        if tx[i, 22] == 0:
            y_jet_zeros.append(y[i])
            x_jet_zeros.append(tx_std[i, :])
            ids_jet_zeros.append(ids[i])
        elif tx[i, 22] == 1:
            y_jet_ones.append(y[i])
            x_jet_ones.append(tx_std[i, :])
            ids_jet_ones.append(ids[i])
        else:
            y_jet_twos.append(y[i])
            x_jet_twos.append(tx_std[i, :])
            ids_jet_twos.append(ids[i])

    x_jet_zeros = np.array(x_jet_zeros)
    x_jet_ones = np.array(x_jet_ones)
    x_jet_twos = np.array(x_jet_twos)
    y_jet_zeros = np.array(y_jet_zeros)
    y_jet_ones = np.array(y_jet_ones)
    y_jet_twos = np.array(y_jet_twos)
    ids_jet_zeros = np.array(ids_jet_zeros)
    ids_jet_ones = np.array(ids_jet_ones)
    ids_jet_twos = np.array(ids_jet_twos)

    print ('The data is now splitted according to the number of jets')
    return x_jet_zeros, x_jet_ones, x_jet_twos, y_jet_zeros, y_jet_ones, y_jet_twos, ids_jet_zeros, ids_jet_ones, ids_jet_twos


def cross_validation(y, tx):
    x1 = np.delete(tx, range(200000, 250000), axis=0)
    x2 = np.delete(tx, range(150000, 200000), axis=0)
    x3 = np.delete(tx, range(100000, 150000), axis=0)
    x4 = np.delete(tx, range(50000, 100000), axis=0)
    x5 = np.delete(tx, range(0, 50000), axis=0)
    y1 = np.delete(y, range(200000, 250000), axis=0)
    y2 = np.delete(y, range(150000, 200000), axis=0)
    y3 = np.delete(y, range(100000, 150000), axis=0)
    y4 = np.delete(y, range(50000, 100000), axis=0)
    y5 = np.delete(y, range(0, 50000), axis=0)
    return x1, x2, x3, x4, x5, y1, y2, y3, y4, y5


def cross_validation_small(y, tx):
    x1 = np.delete(tx, range(8000, 9999), axis=0)
    x2 = np.delete(tx, range(6000, 8000), axis=0)
    x3 = np.delete(tx, range(4000, 6000), axis=0)
    x4 = np.delete(tx, range(2000, 4000), axis=0)
    x5 = np.delete(tx, range(0, 2000), axis=0)
    y1 = np.delete(y, range(8000, 9999), axis=0)
    y2 = np.delete(y, range(6000, 8000), axis=0)
    y3 = np.delete(y, range(4000, 6000), axis=0)
    y4 = np.delete(y, range(2000, 4000), axis=0)
    y5 = np.delete(y, range(0, 2000), axis=0)
    return x1, x2, x3, x4, x5, y1, y2, y3, y4, y5


# We now have to remove the -999 columns that we get since we clustered togehter all the data with specific jets
def remove_useless_column_zeros(tx):
    tx = np.delete(tx, range(23, 29), axis=1)
    tx = np.delete(tx, 12, axis=1)
    tx = np.delete(tx, range(4, 7), axis=1)
    np.savetxt('zeros.txt', tx)

    return tx


# We now have to remove the -999 columns that we get since we clustered togehter all the data with specific jets
def remove_useless_column_ones(tx):
    print (tx.shape)
    tx = np.delete(tx, range(26, 29), axis=1)
    print (tx.shape)
    tx = np.delete(tx, 12, axis=1)
    print (tx.shape)
    tx = np.delete(tx, range(4, 7), axis=1)
    np.savetxt('ones.txt', tx)

    return tx

def PCA(x, threshold):
    """Apply PCA to a given set of datapoints in d-dimension"""
    cov_x = np.cov(x.T)
    eigenValues, eigenVectors = np.linalg.eig(cov_x)
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    eig_val = np.asarray(eigenValues)
    eig_vec = np.asarray(eigenVectors)
    
    eig_val=eig_val/sum(eig_val)
    k=-1
    sum_=0
    while(sum_<threshold):
        k+=1
        sum_=sum_+eig_val[k]
    
    #keep only kth first dimension
    
    eig_vec=eig_vec[:,:k]
    
    return eig_val, eig_vec, k