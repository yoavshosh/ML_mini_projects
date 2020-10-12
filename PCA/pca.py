import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC



def plot_vector_as_image(image, h, w, title):
    """
    utility function to plot a vector as image.
    Args:
    image - vector of pixels
    h, w - dimesnions of original pi
    """	
    plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
    plt.title(title, size=12)
    plt.savefig(title.replace(' ','_')+'.png')
    plt.close()

def get_pictures_by_name(name='Ariel Sharon'):
    """
    Given a name returns all the pictures of the person with this specific name.
    YOU CAN CHANGE THIS FUNCTION!
    THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
    """
    lfw_people = load_data()
    selected_images = []
    n_samples, h, w = lfw_people.images.shape
    target_label = list(lfw_people.target_names).index(name)
    for image, target in zip(lfw_people.images, lfw_people.target):
        if (target == target_label):
            image_vector = image.reshape((h*w, 1))
            selected_images.append(image_vector)
    return np.reshape(selected_images, (len(selected_images),h*w)), h, w

def load_data():
    # Don't change the resize factor!!!
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return lfw_people

######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""

# section a
def pca(X, k):
    """
    Compute PCA on the given matrix.
    
    Args:
        X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
            For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
        k - number of eigenvectors to return

    Returns:
        U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors 
            of the covariance matrix.
        S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
    """
    
    n = np.shape(X)[0]
    d = np.shape(X)[1]
    
    #shift data to have mean=0
    mean = np.average(X, axis = 0)
    X_shifted = [np.array(i) - mean for i in X]
    X_shifted = np.reshape(X_shifted,(n,d))
    
    #singular value decomposition for data
#    M, Sigma_arr, Vt = np.linalg.svd(cov_mat, full_matrices = False)
    M, Sigma_arr, Vt = np.linalg.svd(X_shifted, full_matrices = False)

    S = np.reshape(np.array([i**2 for i in Sigma_arr[:k]]),(min(k,n),1))
    U = Vt[:k,:]
#    S = Sigma_arr[:k]
    
    return U, S




if __name__ == '__main__':
    
    selected_name = 'George W Bush'
#    selected_name = 'Ariel Sharon'
    selected_images,h,w = get_pictures_by_name(name=selected_name)
    
    #section b
    k=10
    print(f'\nRunning PCA for {selected_name} images, and collecting {k} principal eigenpictures')
    U, S = pca(selected_images,10)
    for i, image in enumerate(U):
#        plot_vector_as_image(image,h,w,f'Eigen vec {i}')
        pass
        
    #section c
    print(f'\nRunning PCA for {selected_name} images, with different k values')
    idxs = np.random.choice(range(len(selected_images)), 10, replace=False)
    [plot_vector_as_image(image,h,w,f'Original Image {i}') for i,image in enumerate(selected_images[idxs])]
    k_arr = [1,5,10,30,50,100]
    distances = []
    for k in k_arr:
        U, S = pca(selected_images,k)
#        print(f'K={k}, eigenvalues = {S}')
        distances_norm = 0
        for i,image in enumerate(selected_images[idxs]):
            transformed_pic = np.dot(U.T,np.dot(U,image))
            plot_vector_as_image(transformed_pic,h,w,f'Transformed Image {i} K={k}')
            distances_norm += np.linalg.norm(image - transformed_pic)
        distances.append(distances_norm)
        print(f'K={k}, Distances norm = {round(distances_norm,2)}')
    
# =============================================================================
#     plt.plot(k_arr,distances, marker = '+')
#     plt.xlabel('K')
#     plt.ylabel('Norm of Picture Distances Sum')
#     plt.savefig('L2_norm_wrt_k.png')
#     plt.close()
#     
#     plt.plot(np.arange(1,len(S)+1),S)
#     plt.xlabel('K')
#     plt.xlabel('Eigenvalue')
#     plt.savefig('Eigenvalues_wrt_k.png')
#     plt.close()
# =============================================================================
    
    
    #section d
    #based on a piece of code from
    # https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py
    lfw_people = load_data()
    n_samples, h, w = lfw_people.images.shape    
    X = lfw_people.data
    n_features = X.shape[1]    
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]
    print('\nRunning PCA for many face, with different k values, classifying test data using SVM with GridSearch')
    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    
    param_grid = {'C': [10,5000,10000],
                  'gamma': [1e-7,1e-9,1e-9,1e-10], }
    #performing PCA for different k values
    acc_arr = []
    k_arr = [1,5,10,30,50,100,150,300]
#    k_arr = [50]
    for k in k_arr:
        print("\nExtracting the top %d eigenfaces from %d faces" % (k, X_train.shape[0]))
        
        U ,S = pca(X_train,k)
        X_train_pca = np.array([list(np.dot(U,x)) for x in X_train])
        X_test_pca = np.array([list(np.dot(U,x)) for x in X_test])
        print("Fitting the classifier to the training set")

        clf = GridSearchCV(SVC(kernel = 'rbf', class_weight='balanced'), param_grid, cv=5, iid = True)
        clf = clf.fit(X_train_pca, y_train)
        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
        
        print("\nPredicting people's names on the test set")
        y_pred = clf.predict(X_test_pca)
        
        acc = sum([1 for i,pred in enumerate(y_pred) if pred==y_test[i]])/len(y_pred)
        acc_arr.append(acc)
        print(f'Accuracy = {round(acc,3)}')
        
# =============================================================================
#     plt.plot(k_arr,acc_arr,marker = '+')
#     plt.xlabel('K')
#     plt.ylabel('Accuracy')
#     plt.savefig('acc_wrt_k.png')
#     plt.close()
#     
#     plt.plot(np.arange(1,len(S)+1),S)
#     plt.xlabel('K')
#     plt.xlabel('Eigenvalue')
#     plt.savefig('Eigenvalues_wrt_k_d.png')
#     plt.close()
# =============================================================================
