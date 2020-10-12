#################################
# Your name: Yoav Shoshan
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import fetch_mldata



"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""

def perceptron(data, labels):
    """
    returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
    """
    w = np.zeros(data.shape[1])
    for i in range(data.shape[0]):
        if np.sign(np.dot(data[i],w)) != labels[i]:
            w = w + data[i]*labels[i]
    return w



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

 
#plt.imshow(np.reshape(test_data[1], (28, 28)), interpolation='nearest')

def test_perceptron(w, test_data, test_labels):
    n_bad = 0
    for i in range(test_data.shape[0]):
        if np.sign(np.dot(w,test_data[i])) != test_labels[i]:
            n_bad+=1
    return 1-(n_bad/test_data.shape[0])


def find_misclassifications(w, test_data, test_labels):
    misclassifications_data = []
    misclassifications_labels = []
    misclassifications_indeces = []
    for i in range(test_data.shape[0]):
        if np.sign(np.dot(w,test_data[i])) != test_labels[i]:
            misclassifications_data.append(test_data[i])
            misclassifications_labels.append(test_labels[i])
            misclassifications_indeces.append(i)
    return np.array(misclassifications_data), misclassifications_labels, misclassifications_indeces
  
          
def plot_and_save_perceptron(matrix, image = None):
    fig, ax = plt.subplots()
    plt.imshow(np.reshape(matrix, (28, 28)), interpolation='nearest')
    if image == None:
        plt.title('Perceptron Classifier Image from Perceptron\non Full Train Data Set')
        plt.savefig(f'perceptron_classifier.pdf')
    else:
        plt.title('A misclassified test Image by Perceptron\non Full Train Data Set')
        plt.savefig(f'perceptron_misclassified{image}.pdf')
    plt.close(fig)
    

def unscaled_data():
    
    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = 0, 8
    train_idx = np.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = np.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])
    
    train_data = data[train_idx[:6000], :].astype(float)
    validation_data = data[train_idx[6000:], :].astype(float)
    test_data = data[60000+test_idx, :].astype(float)
    
    return train_data, validation_data, test_data

if __name__ == '__main__':
    
    #generating data set
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()  

    #section a
    n_arr = np.array([5,10,50,100,500,1000,5000])
    T = 100
    n_mean_acc = []
    print(f'\nRunning perceptrion for train data sizes = {n_arr}, {T} times per train data size')
    for n in n_arr:
        normalized_train_data = np.array([v/np.linalg.norm(v) for v in train_data[range(n),:]])
        idxs = list(range(n)) #indeces for shuffling in each iteration and changin input order
        n_acc = []
        for i in range(T):
            np.random.shuffle(idxs)  # change order of input vectors randomaly 
            classifier = perceptron(normalized_train_data[idxs,:], train_labels[idxs])
            n_acc.append(test_perceptron(classifier,test_data, test_labels))
        
        n_mean_acc.append((n,np.mean(n_acc),np.percentile(np.array(n_acc),5),np.percentile(np.array(n_acc),95)))
    
    print('\nAccuracy results on test data:')
    print('N\tAccuracy\t5 percentile\t95 percentile')
    [print(f'{n}\t{round(acc,4)}\t\t{round(per5,4)}\t\t{round(per95,4)}') for n, acc, per5, per95 in n_mean_acc]
    
    #section b
    print(f'\nRunning perceptrion for the full train data set')
    normalized_train_data = np.array([v/np.linalg.norm(v) for v in train_data])
    classifier = perceptron(normalized_train_data, train_labels)
    plot_and_save_perceptron(classifier)
    
    #section c
    accuracy = test_perceptron(classifier,test_data, test_labels)
    print(f'Accuracy of the result classifier on the test data set is {round(accuracy,4)}')
 
    #section d
    #get misclassified data, its labels, and row indeces within test data ndarray
    misclassifications_data, misclassifications_labels, misclassifications_indeces = find_misclassifications(classifier,test_data, test_labels)
    # get unscaled data
    unscaled_train_data, unscaled_validation_data, unscaled_test_data = unscaled_data()
    #plot a misclassified picture    
    plot_and_save_perceptron(unscaled_test_data[misclassifications_indeces[0]],image = 1)  
    plot_and_save_perceptron(unscaled_test_data[misclassifications_indeces[1]],image = 2)
    

    