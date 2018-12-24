"""
This is the best submission of the team "My little Poly".
The code runtime is quite long since many calculations are
performed in the code. To predict one single set of lambda
and gamma hyperparameters.
NOTE: The plots where made on with Jupiter Notebook in order
to access to the plot library
"""
import numpy as np
import matplotlib.pyplot as plt

from implementations import *
from proj1_helpers import *

print("This is the best submission of the team My little Poly. The code runtime is quite long since many calculations are performed in the code. To predict one single set of lambda and gamma hyperparameters.")

print("Loading the train data")
y, tX, ids = load_csv_data('train.csv')

print("Cleanning the missing values and standardizing the data")
# Removing bothering data and centering
standardized_tX, stddev_train, mean_train = standardize_original(tX)

PCA_flag=0 #change flag to 1 to perform PC
if(PCA_flag):
    print("Applying PCA on training dataset")
    eig_val, eig_vec, j = PCA(standardized_tX, threshold=0.7)
    standardized_tX = standardized_tX.dot(eig_vec)
    print(j)
#After PCA the remaining number of dimensions (j) should not be too small for proper polynomial basis contruction (threshold>0.7)

print("Building polynomial basis")
mat_train = build_poly_basis(standardized_tX)

print("Standardizing the matrix contacting the Polynome")
standardized_mat, stddev_poly_train, mean_poly_train = standardize_basis(mat_train)
tx = np.c_[np.ones(standardized_mat.shape[0]), standardized_mat]

print("Training Begins Little Padawan Regression")
print("Please wait while the training completes")
# Cross validating
x1, x2, x3, x4, x5, y1, y2, y3, y4, y5 = cross_validation(y, tx)
#x1, x2, x3, x4, x5, y1, y2, y3, y4, y5 = cross_validation_small(y, tx) #If you want to run the code on a smaller dataset

# Best found parameters for reg_logistic_regression where:
# lambda = 3*10**-9
# gamma  = 10**5

#LAST TESTED VALUES
#lambdas_ = np.linspace(10**-9, 4*10**-9, 4)
#gammas_ = np.linspace(4.5*10**5, 5.5*10**5, 3)

# Best found parameters for reg_logistic_regression
lambdas_ = [3.74*10**-9]
gammas_ = [10**5]
max_iters = 25
initial_w = np.zeros((len(x1[0])))
logistic_weights = []
loss = []
losses = np.zeros((len(gammas_), len(lambdas_)))

for i in range(len(lambdas_)):
    for j in range(len(gammas_)):
        print(lambdas_[i])
        print(gammas_[j])
        reg_logistic_w1,loss1 = reg_logistic_regression(y1,x1,lambdas_[i],initial_w,max_iters,gammas_[j])
        reg_logistic_w2,loss2 = reg_logistic_regression(y2,x2,lambdas_[i],initial_w,max_iters,gammas_[j])
        reg_logistic_w3,loss3 = reg_logistic_regression(y3,x3,lambdas_[i],initial_w,max_iters,gammas_[j])
        reg_logistic_w4,loss4 = reg_logistic_regression(y4,x4,lambdas_[i],initial_w,max_iters,gammas_[j])
        reg_logistic_w5,loss5 = reg_logistic_regression(y5,x5,lambdas_[i],initial_w,max_iters,gammas_[j])
        logistic_weights.append((reg_logistic_w1+reg_logistic_w2+reg_logistic_w3+reg_logistic_w4+reg_logistic_w5)/5)
        losses[i,j] = ((loss1+loss2+loss3+loss4+loss5)/5)
        loss.append((loss1+loss2+loss3+loss4+loss5)/5)

min_losses = np.amin(losses)
idx_min = np.argmin(losses)

print("Loading the testing data")
_, testx, ids_test = load_csv_data('test.csv')

print("Applying the same standardization to the testing set")
standardized_testx = standardize_test_original(testx, mean_train, stddev_train)

if(PCA_flag):
    print("Remove dimensions according to PCA")
    standardized_testx = standardized_testx.dot(eig_vec)

print("Building polynomial basis")
mat_test = build_poly_basis(standardized_testx)

print("Standardizing again")
standardized_testmat = standardized_testx_basis(mat_test, mean_poly_train, stddev_poly_train)
tX_test = np.c_[np.ones(standardized_testmat.shape[0]), standardized_testmat]

logistic_weights_min =  logistic_weights[idx_min]

idx_lambdas = int(idx_min / len(lambdas_))
idx_gammas = idx_min % len(lambdas_)

print("The best submission parameter was [lambda]", lambdas_[idx_lambdas])
print("The best submission parameter was [gamma]", gammas_[idx_gammas])

y_pred = predict_labels(logistic_weights_min, tX_test)
create_csv_submission(ids_test, y_pred, 'predictions_ALL.csv')
print("The prediction has been stored in the predictions.csv file")

#Control the accuracy of the output
y_pred = predict_labels(logistic_weights_min, tx)
counter = 0.0
for i in range (0, len(y)):
    if (y_pred[i] == y[i]):
        counter = counter + 1
print ('Accuracy', counter /len(y))

fig = plt.figure()
# Make data.
X = lambdas_
Y = gammas_
X, Y = np.meshgrid(X, Y)
Z = losses

# Plot the surface.
surf = plt.contourf(X, Y, Z)
# Customize the z axis.
plt.xlabel('Lambda')
plt.ylabel('Gamma')
plt.title('Loss Function for the hyperparameters $lambda$ and $gamma$')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('Surface_plot_LR_jets.png', bbox_inches = 'tight')
plt.show()