"""
This is the best submission of the team "My little Poly".
The code runtime is quite long since many calculations are performed in the code. To predict one single set of
lambda and gamma hyperparameters.
NOTE: The plots where made on with Jupiter Notebook in order to access to the plot library
"""
import numpy as np
import matplotlib.pyplot as plt

from implementations import *
from proj1_helpers import *

print("Loading the train data")
y, tX, ids = load_csv_data('train_small.csv')

print("Cleaning and standardizing the data")
# Removing Missing data and Centering
standardized_tX, stdevtrain, m = standardize_original(tX)
x_jet_zeros_tr, x_jet_ones_tr, x_jet_twos_tr, y_jet_zeros_tr, y_jet_ones_tr, y_jet_twos_tr, ids_jet_zeros_tr, ids_jet_ones_tr, ids_jet_twos_tr = sliptJets(y, tX, standardized_tX, ids)

x_jet_ones_tr = remove_useless_column_ones(x_jet_ones_tr)
x_jet_zeros_tr = remove_useless_column_zeros(x_jet_zeros_tr)

print("Building polynomial basis")
mat_x_jet_zeros_tr = build_poly_basis(x_jet_zeros_tr)
mat_x_jet_ones_tr = build_poly_basis(x_jet_ones_tr)
mat_x_jet_twos_tr = build_poly_basis(x_jet_twos_tr)

print("Standardizing The Training set's Polynomial Basis")
mat_x_jet_zeros_tr, stdev_mat_x_jet_zeros_tr, m_mat_x_jet_zeros_tr = standardize_basis(mat_x_jet_zeros_tr)
mat_x_jet_ones_tr, stdev_mat_x_jet_ones_tr, m_mat_x_jet_ones_tr = standardize_basis(mat_x_jet_ones_tr)
max_x_jet_twos_tr, stdev_mat_jet_twos_tr, m_max_jet_twos_tr = standardize_basis(mat_x_jet_twos_tr)

mat_x_jet_zeros_tr = np.c_[np.ones(mat_x_jet_zeros_tr.shape[0]), mat_x_jet_zeros_tr]
mat_x_jet_ones_tr = np.c_[np.ones(mat_x_jet_ones_tr.shape[0]), mat_x_jet_ones_tr]
mat_x_jet_twos_tr = np.c_[np.ones(mat_x_jet_twos_tr.shape[0]), mat_x_jet_twos_tr]

print("Training...")

### CROSS VALIDATION ###
'''For the small data sample of 10 000 observsation for 30 features'''
x1_zeros, x2_zeros, x3_zeros, x4_zeros, x5_zeros, y1_zeros, y2_zeros, y3_zeros, y4_zeros, y5_zeros = cross_validation_prep_zeros_small(y_jet_zeros_tr, mat_x_jet_zeros_tr)
x1_ones, x2_ones, x3_ones, x4_ones, x5_ones, y1_ones, y2_ones, y3_ones, y4_ones, y5_ones = cross_validation_prep_ones_small(y_jet_ones_tr, mat_x_jet_ones_tr)
x1_twos, x2_twos, x3_twos, x4_twos, x5_twos, y1_twos, y2_twos, y3_twos, y4_twos, y5_twos = cross_validation_prep_twos_small(y_jet_twos_tr, mat_x_jet_twos_tr)

'''For the entire train data set'''
#x1_zeros, x2_zeros, x3_zeros, x4_zeros, x5_zeros, y1_zeros, y2_zeros, y3_zeros, y4_zeros, y5_zeros = cross_validation_prep_zeros(y_jet_zeros_tr, mat_x_jet_zeros_tr)
#x1_ones, x2_ones, x3_ones, x4_ones, x5_ones, y1_ones, y2_ones, y3_ones, y4_ones, y5_ones = cross_validation_prep_ones(y_jet_ones_tr, mat_x_jet_ones_tr)
#x1_twos, x2_twos, x3_twos, x4_twos, x5_twos, y1_twos, y2_twos, y3_twos, y4_twos, y5_twos = cross_validation_prep_twos(y_jet_twos_tr, mat_x_jet_twos_tr)

#Best found parameters for reg_logistic_regression
#gammas_ = np.linspace(10**5)
lambdas_ = np.linspace(10**-10, 1, 20)
#np.array([10**-9, 10**-8,10**-7,10**-6, 10**-5])

gammas_ = np.linspace(10**4, 10**5, 20)
#lambdas_ = np.array([10**-8])  #, 10**-9

max_iters = 25

losses_zeros = np.zeros((len(gammas_), len(lambdas_)))
weights_zeros = []
losses_ones = np.zeros((len(gammas_), len(lambdas_)))
weights_ones = []
losses_twos = np.zeros((len(gammas_), len(lambdas_)))
weights_twos = []

for i in range (0, len(gammas_)):
    for j in range(0, len(lambdas_)):
        print("LAMBDAS_: ",i)
        print("GAMMAS_:", j)

        initial_w = np.zeros((len(x1_zeros[0])))
        reg_logistic_w1_zeros,loss_1_zeros = reg_logistic_regression(y1_zeros,x1_zeros,lambdas_[j],initial_w,max_iters,gammas_[i])
        initial_w = np.zeros((len(x2_zeros[0])))
        reg_logistic_w2_zeros,loss_2_zeros = reg_logistic_regression(y2_zeros,x2_zeros,lambdas_[j],initial_w,max_iters,gammas_[i])
        initial_w = np.zeros((len(x3_zeros[0])))
        reg_logistic_w3_zeros,loss_3_zeros = reg_logistic_regression(y3_zeros,x3_zeros,lambdas_[j],initial_w,max_iters,gammas_[i])
        initial_w = np.zeros((len(x4_zeros[0])))
        reg_logistic_w4_zeros,loss_4_zeros = reg_logistic_regression(y4_zeros,x4_zeros,lambdas_[j],initial_w,max_iters,gammas_[i])
        initial_w = np.zeros((len(x5_zeros[0])))
        reg_logistic_w5_zeros,loss_5_zeros = reg_logistic_regression(y5_zeros,x5_zeros,lambdas_[j],initial_w,max_iters,gammas_[i])
        logistic_weights_zeros = (reg_logistic_w1_zeros+reg_logistic_w2_zeros+reg_logistic_w3_zeros+reg_logistic_w4_zeros+reg_logistic_w5_zeros)/5

        weights_zeros.append(logistic_weights_zeros)
        losses_zeros[i,j] = ((loss_1_zeros+loss_2_zeros+loss_3_zeros+loss_4_zeros+loss_5_zeros)/5)

        initial_w = np.zeros(len(x1_ones[0]))
        reg_logistic_w1_ones,loss_1_ones = reg_logistic_regression(y1_ones,x1_ones,lambdas_[j],initial_w,max_iters,gammas_[i])
        initial_w = np.zeros(len(x2_ones[0]))
        reg_logistic_w2_ones,loss_2_ones = reg_logistic_regression(y2_ones,x2_ones,lambdas_[j],initial_w,max_iters,gammas_[i])
        initial_w = np.zeros((len(x3_ones[0])))
        reg_logistic_w3_ones,loss_3_ones = reg_logistic_regression(y3_ones,x3_ones,lambdas_[j],initial_w,max_iters,gammas_[i])
        initial_w = np.zeros((len(x4_ones[0])))
        reg_logistic_w4_ones,loss_4_ones = reg_logistic_regression(y4_ones,x4_ones,lambdas_[j],initial_w,max_iters,gammas_[i])
        initial_w = np.zeros((len(x5_ones[0])))
        reg_logistic_w5_ones,loss_5_ones = reg_logistic_regression(y5_ones,x5_ones,lambdas_[j],initial_w,max_iters,gammas_[i])
        logistic_weights_ones = (reg_logistic_w1_ones+reg_logistic_w2_ones+reg_logistic_w3_ones+reg_logistic_w4_ones+reg_logistic_w5_ones)/5

        weights_ones.append(logistic_weights_ones)
        losses_ones[i,j] = ((loss_1_ones + loss_2_ones + loss_3_ones + loss_4_ones + loss_5_ones) / 5)

        initial_w = np.zeros(len(x1_twos[0]))
        print(x1_twos.shape[1])
        reg_logistic_w1_twos,loss_1_twos = reg_logistic_regression(y1_twos,x1_twos,lambdas_[j],initial_w,max_iters,gammas_[i])
        initial_w = np.zeros(len(x2_twos[0]))
        reg_logistic_w2_twos,loss_2_twos = reg_logistic_regression(y2_twos,x2_twos,lambdas_[j],initial_w,max_iters,gammas_[i])
        initial_w = np.zeros(len(x3_twos[0]))
        reg_logistic_w3_twos,loss_3_twos = reg_logistic_regression(y3_twos,x3_twos,lambdas_[j],initial_w,max_iters,gammas_[i])
        initial_w = np.zeros(len(x4_twos[0]))
        reg_logistic_w4_twos,loss_4_twos = reg_logistic_regression(y4_twos,x4_twos,lambdas_[j],initial_w,max_iters,gammas_[i])
        initial_w = np.zeros(len(x5_twos[0]))
        reg_logistic_w5_twos,loss_5_twos = reg_logistic_regression(y5_twos,x5_twos,lambdas_[j],initial_w,max_iters,gammas_[i])
        logistic_weights_twos = (reg_logistic_w1_twos+reg_logistic_w2_twos+reg_logistic_w3_twos+reg_logistic_w4_twos+reg_logistic_w5_twos)/5

        weights_twos.append(logistic_weights_twos)
        losses_twos[i,j] = ((loss_1_twos + loss_2_twos + loss_3_twos + loss_4_twos + loss_5_twos) / 5)

print ('zeros', np.amin(losses_zeros))
print ('ones', np.amin(losses_ones))
print ('twos', np.amin(losses_twos))

np.savetxt('losses_zeros.txt', losses_zeros)
np.savetxt('losses_ones.txt', losses_ones)
np.savetxt('losses_twos.txt', losses_twos)

print("Loading the testing data")
testy, testx, ids_test = load_csv_data('test_small.csv')

print("Applying the same standardization to the testing set")
standardized_testx = standardize_test_original(testx, m, stdevtrain)
x_jet_zeros_te, x_jet_ones_te, x_jet_twos_te, y_jet_zeros_te, y_jet_ones_te, y_jet_twos_te, ids_jet_zeros_te, ids_jet_ones_te, ids_jet_twos_te = sliptJets(testy, testx, standardized_testx, ids_test)

x_jet_ones_te = remove_useless_column_ones(x_jet_ones_te)
x_jet_zeros_te = remove_useless_column_zeros(x_jet_zeros_te)

print("Building polynomial basis")
"""
use the same polynomial basis as before.
"""
mat_x_jet_zeros_te = build_poly_basis(x_jet_zeros_te)
mat_x_jet_ones_te = build_poly_basis(x_jet_ones_te)
mat_x_jet_twos_te = build_poly_basis(x_jet_twos_te)

print("Standardizing the jets")
mat_x_jet_zeros_te = standardized_testx_basis(mat_x_jet_zeros_te, m_mat_x_jet_zeros_tr, stdev_mat_x_jet_zeros_tr)
mat_x_jet_ones_te = standardized_testx_basis(mat_x_jet_ones_te, m_mat_x_jet_ones_tr, stdev_mat_x_jet_ones_tr)
mat_x_jet_twos_te = standardized_testx_basis(mat_x_jet_twos_te, m_max_jet_twos_tr, stdev_mat_jet_twos_tr)

mat_x_jet_zeros_te = np.c_[np.ones(mat_x_jet_zeros_te.shape[0]), mat_x_jet_zeros_te]
mat_x_jet_ones_te = np.c_[np.ones(mat_x_jet_ones_te.shape[0]), mat_x_jet_ones_te]
mat_x_jet_twos_te = np.c_[np.ones(mat_x_jet_twos_te.shape[0]), mat_x_jet_twos_te]

'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

################################ 2D Contour plot ###################################
fig = plt.figure()
# Make data.
X = lambdas_
Y = gammas_
X, Y = np.meshgrid(X, Y)
Z_zeros = losses_zeros
Z_ones = losses_ones
Z_twos = losses_twos
# Plot the surface.
surf = plt.contourf(X, Y, Z_zeros)
# Customize the z axis.
plt.xlabel('Lambda')
plt.ylabel('Gamma')
plt.title('Loss Function for the hyperparameters $lambda$ and $gamma$')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('Contour_plot_LR_2D_ZEROS.png', bbox_inches = 'tight')
plt.show()
i = np.floor(np.argmin(losses_zeros)/len(gammas_))
j = np.argmin(losses_zeros)%len(gammas_)
i = int(i)
print('ZEROS: The best gamma is: ', gammas_[j])
print('ZEROS: The best lambda is: ', lambdas_[i])


fig = plt.figure()
# Make data.
X = lambdas_
Y = gammas_
X, Y = np.meshgrid(X, Y)
# Plot the surface.
surf = plt.contourf(X, Y, Z_ones)
# Customize the z axis.
plt.xlabel('Lambda')
plt.ylabel('Gamma')
plt.title('Loss Function for the hyperparameters $lambda$ and $gamma$')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('Contour_plot_LR_2D_ONES.png', bbox_inches = 'tight')
plt.show()
i = np.floor(np.argmin(losses_ones)/len(gammas_))
j = np.argmin(losses_ones)%len(gammas_)
i = int(i)
print('ONES: The best gamma is: ', gammas_[j])
print('ONES: The best lambda is: ', lambdas_[i])

fig = plt.figure()
# Make data.
X = lambdas_
Y = gammas_
X, Y = np.meshgrid(X, Y)
# Plot the surface.
surf = plt.contourf(X, Y, Z_twos)
# Customize the z axis.
plt.xlabel('Lambda')
plt.ylabel('Gamma')
plt.title('Loss Function for the hyperparameters $lambda$ and $gamma$')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('Contour_plot_LR_2D_TWOS.png', bbox_inches = 'tight')
plt.show()
i = np.floor(np.argmin(losses_zeros)/len(gammas_))
j = np.argmin(losses_zeros)%len(gammas_)
i = int(i)
print('TWOS: The best gamma is: ', gammas_[j])
print('TWOS: The best lambda is: ', lambdas_[i])

y_pred_zeros = predict_labels(weights_zeros[np.argmin(losses_zeros)], mat_x_jet_zeros_te)
y_pred_ones = predict_labels(weights_ones[np.argmin(losses_ones)], mat_x_jet_ones_te)
y_pred_twos = predict_labels(weights_twos[np.argmin(losses_twos)], mat_x_jet_twos_te)

#Conbines the 3 jet vectors
ids = np.append(ids_jet_zeros_te, ids_jet_ones_te)
ids = np.append(ids, ids_jet_twos_te)

LR_y = np.append(y_pred_zeros, y_pred_ones)
LR_y = np.append(LR_y, y_pred_twos)

create_csv_submission(ids_jet_zeros_te, y_pred_zeros, 'predictions_jets_poly.csv')
print("The prediction has been stored in the predictions.csv file")

#Control the accuracy of the prediction
print("Control the accuracy of the prediction")
y_pred_zeros = predict_labels(logistic_weights_zeros, mat_x_jet_zeros_tr)
y_pred_ones = predict_labels(logistic_weights_ones, mat_x_jet_ones_tr)
y_pred_twos = predict_labels(logistic_weights_twos, mat_x_jet_twos_tr)
counter = 0.0
for i in range (0, len(y_pred_zeros)):
    if (y_pred_zeros[i] == y_jet_zeros_tr[i]):
        counter = counter + 1
print ('zeros', counter /len(y_pred_zeros))

counter = 0.0
for i in range (0, len(y_pred_ones)):
    if (y_pred_ones[i] == y_jet_ones_tr[i]):
        counter = counter + 1
print ('ones', counter /len(y_pred_ones))
counter = 0.0
for i in range (0, len(y_pred_twos)):
    if (y_pred_twos[i] == y_jet_twos_tr[i]):
        counter = counter + 1
print ('twos', counter /len(y_pred_twos))