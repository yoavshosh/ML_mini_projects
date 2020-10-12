import os
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import math
from sklearn.datasets import fetch_mldata
from scipy import spatial
from collections import Counter


"""
returns a list of most common values in an input list
output size is as number of elements sharing the max count value
this function is called by knn function at the end of knn process, to get a list of most frequenct labels among knn.
"""
def most_common(lst):
    max_cnt = None
    common_vals = []
    for label, cnt in Counter(lst).most_common(len(set(lst))):  #Counter(lst).most_common retrive pairs of item:count srted by count, sorted from largest to smallest count
        if cnt==max_cnt or max_cnt == None: 
            common_vals.append(label)
            max_cnt = cnt
        else:
            break
    return common_vals
            

"""
returns a prediction of query label usin knn algorithem
arguments:
neighbors_set - a list of al neighbors vectors (vectors of pixle values)
neighbors_labels - corresponding labels for neighbors_set
query - pixles vector of qurey image
k - number of knn based on wich prediction is made
"""
def knn(neighbors_set, neighbors_labels, query, k):
    dists = spatial.distance.cdist(neighbors_set, np.array([query]), metric = 'euclidean') #calculate distances between neighbors_set and query
    indeces_sorted_by_dist = [x for _,x in sorted(zip(dists,np.arange(len(neighbors_set))))] #sort neighbors indeces by neighbors distances from query
    knn_labels = neighbors_labels[indeces_sorted_by_dist[:k]] #retrive labels of knn 
    most_common_labels = most_common(knn_labels) #retrive a list of most common labels from knn labels
    return np.random.choice(most_common_labels) #return an arbitrary labels from knn most common label(s)
    


"""
run knn for multiple test images using a given train set
return accuracy of prediction using 0/1 loss function
"""
def run_knn_for_multiple_test_images(train_set, train_labels, test_set, test_labels, k):
    
    predicted_labels = []
    for t in test_set:
        predicted_labels.append(knn(train_set,train_labels,t,k))
    loss_vals = [0 if tl==pl else 1 for tl,pl in zip(test_labels,predicted_labels)]
    tot_loss = sum(loss_vals)
    accuracy_knn = round((len(test_labels)-tot_loss)/len(test_labels),3)
    
    return tot_loss,accuracy_knn



"""
Plot Accuracy of prediction with respect to K in KNN results
arguments:
x - K array
y - accuracies array
"""
def plot_acc_vs_k(x,y):

    o_path = os.getcwd()
    fig, ax = plt.subplots()
    plt.xlabel('K')
    plt.ylabel('Accuracy of Prediction')
    plt.title('Accuracy of Prediction in KNN Algorithm with Respect to K')
    plt.scatter(x, y)
    plt.xlim(0,max(x))
    plt.ylim(max(0,math.floor(min(y)*10)/10),1.05)
    plt.xticks(np.linspace(0,len(x),6))
    plt.yticks(np.arange(max(0,math.floor(min(y)*10)/10),1.05,0.1))
    plt.savefig(o_path+'/Accuracy_vs_K.pdf')
    print('Accuracy_vs_K plot saved to CWD')
    
    
"""
Plot Accuracy of prediction with respect to train data size in KNN results
arguments:
x - K array
y - accuracies array
k - k value used for KNN algorithm
"""
def plot_acc_vs_n(x,y,k):

    o_path = os.getcwd()
    fig, ax = plt.subplots()
    plt.xlabel('N')
    plt.ylabel('Accuracy of Prediction')
    plt.title(f'Accuracy of Prediction in KNN Algorithm (K = {k})\nwith Respect to Train Data Size')
    plt.scatter(x, y)
    plt.xlim(0,max(x))
    plt.ylim(max(0,math.floor(min(y)*10)/10),1.05)
    plt.xticks(np.linspace(0,max(x),6))
    plt.yticks(np.arange(max(0,math.floor(min(y)*10)/10),1.05,0.1))
    plt.savefig(o_path+'/Accuracy_vs_N.pdf')
    print('Accuracy_vs_N plot saved to CWD')

    
if __name__ == '__main__':
    
    #loading data
    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    labels = mnist['target']
    
    #defining train data and test data
    idx = np.random.RandomState(0).choice(70000, 11000)
    train = data[idx[:10000], :].astype(int)
    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]]
    
    #scetion b. running KNN for a given train data, test data and K. 
    k= 10
    N = 1000
    print(f'\nRunning KNN for {len(test_labels)} test images. train set size = {N}, K = {k}')
    tot_loss,accuracy_knn = run_knn_for_multiple_test_images(train[:N],train_labels[:N],test,test_labels,k)
    print(f'Accumulated loss (using 1/0 loss function) is {tot_loss}')
    print(f'Accuracy using knn is {accuracy_knn}, Accuracy using random predictor is {0.1} (there are 10 possible labels to choose from)')
    
    #section c. running KNN for a given train data, test data and an increasing K.
    kstart = 1
    kstop = 100
    k_arr = np.arange(kstart,kstop+1)
    N = 1000
    print(f'\nRunning KNN for {len(test_labels)} test images. train set size = {N}, K from {k_arr[0]} to {k_arr[-1]}')
    acc_list = []
    for k in k_arr:
        _,accuracy_knn = run_knn_for_multiple_test_images(train[:N],train_labels[:N],test,test_labels,k)
        acc_list.append(accuracy_knn) 
    plot_acc_vs_k(k_arr,np.array(acc_list)) #plot (and save) prediction accuracy with respect to K
    k_arr_sorted_by_acc  = [x for _,x in sorted(zip(acc_list,k_arr))] #finding the best K (the K corresponding to the highest accuracy)
    print(f'Best K is {k_arr_sorted_by_acc[-1]}')
    
    #section d. running KNN for K=1, a given test data, and an increasing size of train data.
    k=1
    N_start = 100
    N_stop = 5000
    N_jump = 100
    N_arr = np.arange(N_start,N_stop+1,N_jump)
    print(f'\nRunning KNN for {len(test_labels)} test images with K = {k}, and train set size (N) from {N_arr[0]} to {N_arr[-1]}')
    acc_list = []
    for N in N_arr:
        _,accuracy_knn = run_knn_for_multiple_test_images(train[:N],train_labels[:N],test,test_labels,k)
        acc_list.append(accuracy_knn)
    plot_acc_vs_n(N_arr,acc_list,k)
    N_arr_sorted_by_acc = [x for _,x in sorted(zip(acc_list,N_arr))] #finding the best N (the N corresponding to the highest accuracy)
    print(f'Best N is {N_arr_sorted_by_acc[-1]}')
    
    
    


