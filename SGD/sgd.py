#################################
# Your name: Yoav Shoshan
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing


"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""

def helper():
    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    labels = mnist['target']
    
    neg, pos = 0,8
    train_idx = np.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = np.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])
    
    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos)*2-1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos)*2-1

    test_data_unscaled = data[60000+test_idx, :].astype(float)
    test_labels = (labels[60000+test_idx] == pos)*2-1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def SGD(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the final classifier
    """
    w = np.zeros(data.shape[1])
    for t in range(1,T+1):
        eta_t = eta_0/t
        ind = np.random.choice(data.shape[0])
        if labels[ind]*np.dot(w,data[ind]) < 1:
            w = (1-eta_t)*w + C*eta_t*labels[ind]*data[ind]
        else:
            w = (1-eta_t)*w
    return w

    
def test_SGD_classifier(w, data, labels):
    n_bad = 0
    for i in range(data.shape[0]):
        prediction = -1
        if np.dot(w,data[i])>=0:
            prediction = 1
        if prediction != labels[i]:
            n_bad+=1        
    return 1-(n_bad/data.shape[0])


def plot_acc_wrt_something(eta_arr,acc_arr,x_label):
    fig, ax = plt.subplots()
    plt.plot(eta_arr,acc_arr)
    ax.set_xscale('log')
    plt.xlabel(x_label)
    plt.ylabel('Accuracy')
    plt.savefig('acc_wrt_'+x_label+'.pdf')
    plt.close(fig)
    

def plot_and_save_classifier(matrix):
    fig, ax = plt.subplots()
    plt.imshow(np.reshape(matrix, (28, 28)), interpolation='nearest')
    plt.title('SGD classifier Image')
    plt.savefig(f'SGD_classifier.pdf')
    plt.close(fig)

        


if __name__ == '__main__':
    
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    
    #section a
    powers = np.arange(-5,4.1,0.1)
    eta_arr = [float(10**p) for p in powers]
    acc_arr = []
    T = 1000
    C = 1
    iteration_per_eta = 10
    print(f'\nRunning SGD for deifferent values of eta_0, T = {T}, C = {C}, iterations per eta = {iteration_per_eta}\n')
    for eta in eta_arr:
        eta_acc = []
        for i in range(iteration_per_eta):
            classifier = SGD(train_data, train_labels,C,eta,T)
#            print(classifier)
            eta_acc.append(test_SGD_classifier(classifier,validation_data,validation_labels))
        acc_arr.append(np.mean(eta_acc))
    plot_acc_wrt_something(eta_arr,acc_arr,'eta')
    
    best_eta = [x for _,x in sorted(zip(acc_arr,eta_arr))][-1]
    print(f'\nBest eta is {round(best_eta,3)}\n')
    
    #section b
    powers = np.arange(-5,6.1,0.1)
    C_arr = [10.0**p for p in powers]
    acc_arr = []
    T = 1000
    iteration_per_C = 10
    print(f'\nRunning SGD for different values of C, T = {T}, eta = {round(best_eta,3)}, iterations per C = {iteration_per_C}\n')
    for C in C_arr:
        C_acc = []
        for i in range(iteration_per_C):
            classifier = SGD(train_data, train_labels,C,best_eta,T)
            C_acc.append(test_SGD_classifier(classifier,validation_data,validation_labels))
        acc_arr.append(np.mean(C_acc))
    plot_acc_wrt_something(C_arr,acc_arr,'C')
    
    best_C = [x for _,x in sorted(zip(acc_arr,C_arr))][-1]
    print(f'Best C is {round(best_C,5)}\n')
    
    #section c
    T=20000
    print(f'\nRunning SGD for C = {round(best_C,5)}, T = {T}, eta = {round(best_eta,3)}\n')
    classifier = SGD(train_data, train_labels,best_C,best_eta,T)
    plot_and_save_classifier(classifier)
    
    #section d
    accuracy = test_SGD_classifier(classifier,test_data,test_labels)
    print(f'Accuracy of best classifier is {round(accuracy,3)}')

    