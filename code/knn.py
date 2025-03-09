import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss and triangle kernel
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        distances = self.compute_distances(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances(self, X):
        """
        Computes L1 distance from every sample of X to every training sample

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        dist_matr = np.zeros((X.shape[0], self.train_X.shape[0]))
        for i in range(self.train_X.shape[0]):
            for j in range(X.shape[0]):
                dist_matr[j][i] = np.sum(np.abs(self.train_X[i] - X[j]))
        return dist_matr


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)

        for i in range(n_test):
            k_nearest_indexes = np.argsort(distances[i])[:self.k]
            targets = self.train_y[k_nearest_indexes]
            prediction[i] = np.bincount(targets).argmax()
        return prediction


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[0]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, dtype = 'int64')
        for i in range(n_test):
            k_nearest_indexes = np.argsort(distances[i])[:self.k]
            targets = self.train_y[k_nearest_indexes]
            a, b = np.unique(targets, return_counts=True)
            prediction[i] = a[b.argmax()]
        return prediction
