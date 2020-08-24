import numpy as np 
from src.distances import euclidean_distances, manhattan_distances, cosine_distances


class KNearestNeighbor():    
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'manhattan', use manhattan_distances,
        if  'cosine', use cosine_distances.

        ```aggregator``` lets you alter how a label is predicted for a data point based 
        on its neighbors. If it's set to `mean`, it is the mean of the labels of the
        neighbors. If it's set to `mode`, it is the mode of the labels of the neighbors.
        If it is set to median, it is the median of the labels of the neighbors. If the
        number of dimensions returned in the label is more than 1, the aggregator is
        applied to each dimension independently. For example, if the labels of 3 
        closest neighbors are:
            [
                [1, 2, 3], 
                [2, 3, 4], 
                [3, 4, 5]
            ] 
        And the aggregator is 'mean', applied along each dimension, this will return for 
        that point:
            [
                [2, 3, 4]
            ]

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean', 'manhattan', or 'cosine'. This is the distance measure
                that will be used to compare features to produce labels. 
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator
        self.features = []
        self.targets = []



    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class 
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional! 
        
        HINT: One use case of KNN is for imputation, where the features and the targets 
        are the same. See tests/test_collaborative_filtering for an example of this.
        
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples, 
                n_dimensions).
        """

        self.features = features
        self.targets = targets

    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the 
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, the predicted labels are the imputed testing data
        and the shape is (n_samples, n_features).

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor. 
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_dimensions). This n_dimensions should be the same as n_dimensions of targets in fit function.
        """

        labels = np.zeros([features.shape[0], self.targets.shape[1]])

        if self.distance_measure == 'manhattan':
            dis = manhattan_distances(features, self.features)
        elif self.distance_measure == 'cosine':
            dis = cosine_distances(features, self.features)
        else:
            dis = euclidean_distances(features, self.features)

        for i in range(dis.shape[0]):
            # If ignore first, do this
            if ignore_first:
                index = np.argsort(dis[i])[1:(self.n_neighbors + 1)]
            else:
                index = np.argsort(dis[i])
                # if i in index:
                    # index = np.delete(index, np.where(index == i))
                    # index = np.concatenate(([i], index))
                index = index[:self.n_neighbors]
            selected = self.targets[index]

            if self.aggregator == 'mode':
                temp = np.zeros(selected.shape[1])
                # Look at each column in selected targets
                for k in range(selected.shape[1]):
                    selected_col = selected[:, k]
                    # This find the unique values in selected_col and get their counts
                    [val, freq] = np.unique(selected_col, return_counts=True)
                    # Record the unique value that appears most often
                    temp[k] = val[np.argmax(freq)]
                labels[i] = temp

            if self.aggregator == 'mean':
                # Get mean of each column so across row axis
                labels[i] = np.mean(selected, axis=0)

            if self.aggregator == 'median':
                labels[i] = np.median(selected, axis=0)

        return labels

