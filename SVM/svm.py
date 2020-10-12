#################################
# Your name: Yoav Shoshan
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """

    #liniar kernel
    linear_clf = svm.SVC(C=1000, kernel='linear')
    linear_clf.fit(X_train,y_train)
    n_linear_support_v = linear_clf.n_support_
    create_plot(X_train,y_train,linear_clf)
    plt.savefig('Linear_svm.png')
    plt.close()
    
    #quadratic_kernel
    quadratic_clf = svm.SVC(C=1000, kernel='poly', degree = 2)
    quadratic_clf.fit(X_train,y_train)
    n_quadratic_support_v = quadratic_clf.n_support_
    create_plot(X_train,y_train,quadratic_clf)
    plt.savefig('Quadratic_svm.png')
    plt.close()
    
    #rbf kernel
    rbf_clf = svm.SVC(C=1000, kernel='rbf')
    rbf_clf.fit(X_train,y_train)
    n_rbf_support_v = rbf_clf.n_support_
    create_plot(X_train,y_train,rbf_clf)
    plt.savefig('RBF_svm.png')
    plt.close()
    
    return np.vstack((n_linear_support_v,n_quadratic_support_v,n_rbf_support_v))



def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    powers = np.arange(-5,6,1)
    c_arr = [float(10)**p for p in powers]
    acc_arr = []
    best_acc = 0.0
    best_c = 0.0
    
    worst_c = 0
    worst_acc = 1.0
    for c in c_arr:
        linear_clf = svm.SVC(C=c, kernel='linear')
        linear_clf.fit(X_train,y_train)
        acc = linear_clf.score(X_val,y_val)
#        mistakes = 0
#        for i in range(len(y_val)):
#            if linear_clf.predict([X_val[i]]) != y_val[i]:
#                mistakes+=1
#        
#        acc = 1-float(mistakes)/len(y_val)
        
        #determine  best and worst c and their accuracies
        acc_arr.append(acc)
        if acc>best_acc:
            best_acc=acc
            best_c = c
        if acc<worst_acc:
            worst_acc = acc
            worst_c = c
            
    plot_acc_wrt_something(c_arr,acc_arr,"C",'Accuracy on Validation set\nwrt to C')
    
    print(f'Best C is {round(best_c,5)}, with Accuracy = {round(best_acc,20)}\n(if many c values yield the maximal accuracy, this is the smallest among them)')
    
#    best_C_svm = linear_clf = svm.SVC(C=best_c, kernel='linear')
#    best_C_svm.fit(X_train, y_train)
#    create_plot(X_train, y_train, best_C_svm)
#    plt.savefig('Best_C_SVM.png')
#    plt.close()
#    
#    worst_C_svm = linear_clf = svm.SVC(C=worst_c, kernel='linear')
#    worst_C_svm.fit(X_train, y_train)
#    create_plot(X_train, y_train, worst_C_svm)
#    plt.savefig('Worst_C_SVM.png')
#    plt.close()
#    
#    medium_C_svm = linear_clf = svm.SVC(C=0.01, kernel='linear')
#    medium_C_svm.fit(X_train, y_train)
#    create_plot(X_train, y_train, medium_C_svm)
#    plt.savefig('medium_C_SVM.png')
#    plt.close()
    
    return np.array(acc_arr)
            

def plot_acc_wrt_something(x,acc_arr,x_label,title):
    fig, ax = plt.subplots()
    plt.plot(x,acc_arr)
    ax.set_xscale('log')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('Accuracy')
    plt.savefig('acc_wrt_'+x_label+'.png')
    plt.close(fig)
    
    

def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    powers = np.arange(-5,6,1)
    gamma_arr = [float(10)**p for p in powers]
    
    acc_train_arr = []
    acc_validation_arr = []
    
    best_gamma = 0
    best_acc = 0
    for gamma in gamma_arr:
        rbf_clf = svm.SVC(C=10, kernel='rbf', gamma = gamma)
        rbf_clf.fit(X_train,y_train)
        
        acc_train = rbf_clf.score(X_train, y_train)
        acc_train_arr.append(acc_train)
        acc_validatioin = rbf_clf.score(X_val,y_val)
        acc_validation_arr.append(acc_validatioin)
#        create_plot(X_train, y_train, rbf_clf)
#        plt.savefig(str(gamma)+'.png')
#        plt.close()
        
        if acc_validatioin>best_acc:
            best_acc = acc_validatioin
            best_gamma = gamma
            
    
    print(f'Best Gamma on validation set is {round(best_gamma,4)} with Accuracy = {round(best_acc,4)}')
    
    fig, ax = plt.subplots()
    plt.plot(gamma_arr,acc_train_arr)
    plt.plot(gamma_arr,acc_validation_arr)
    ax.set_xscale('log')
    plt.xlabel('gamma')
    plt.ylabel('Accuracy')
    plt.legend(['tain data', 'validation data'], loc='best')
    plt.savefig('Acc_wrt_gamma.png')
    plt.close(fig)
    
    return np.array(acc_validation_arr)
        
        


if __name__ == "__main__":
    
    training_data, training_labels, validation_data, validation_labels = get_points()
    
    #section a
    print('Section a')
    n_support_for_three_kernels = train_three_kernels(training_data, training_labels, validation_data, validation_labels)
    print(f'number of support fectors for linear kernel = {sum(n_support_for_three_kernels[0])}')
    print(f'number of support fectors for quadratic kernel = {sum(n_support_for_three_kernels[1])}')   
    print(f'number of support fectors for rbf kernel = {sum(n_support_for_three_kernels[2])}')
    
    #section b
    print('\nSection b')
    acc_arr_c = linear_accuracy_per_C(training_data, training_labels, validation_data, validation_labels)
    
    #section c
    print('\nSection c')
    acc_arr_gamma = rbf_accuracy_per_gamma(training_data, training_labels, validation_data, validation_labels)
    
    
    
    
    