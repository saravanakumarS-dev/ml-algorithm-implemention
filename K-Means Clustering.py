import numpy as np
import matplotlib.pyplot as plt


class KMeansClustering:

    def __init__(self, k):
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum(np.square(centroids - data_point), axis=1))

    def fit(self, X, max_iterations):


        for _ in range(max_iterations):
            self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))

            Y = np.array([])
            for datapoint in X:
                distance = self.euclidean_distance(datapoint, self.centroids)
                cluster_num = distance.argmin()
                Y = np.append(Y, cluster_num)


            cluster_indices = []
            for i in range(self.k):
                cluster_indices.append(np.argwhere(Y == i).flatten())


            cluster_centers = []
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0))

            if np.max(self.centroids - np.array(cluster_centers)) < 0.001:
                break
            else:
                self.centroids = np.array(cluster_centers)

        return Y

random_points = np.random.randint(0, 100, (100, 2))

kmeans = KMeansClustering(3)
labels = kmeans.fit(random_points, max_iterations=200)

plt.scatter(random_points[:,0], random_points[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)), marker="*", s=200)
plt.show()
