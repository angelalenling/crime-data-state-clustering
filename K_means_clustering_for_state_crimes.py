# crime-data-state-clustering
# K_means_clustering_for_state_crimes.py
# Performs K-Means clustering on the USArrests dataset, calculates crime rate means
# per cluster, and visualizes the results on a color-coded interactive map using Folium.

import pandas as pd
import numpy as np
import folium
import json
import random
import requests
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict


# K-means clustering functions

# computes the squared euclidean distance between vectors x and y
def distance(x, y):
    return np.sum((x - y) ** 2)


# computes the centroid of a cluster (average position of all points in cluster)
# X is a matrix that contains the cluster,
# with rows as observations and (numeric) columns as attributes
def centroid(X):
    return np.mean(X, axis=0)


# a single iteration of kmeans
# X is a matrix of observations with m rows and n columns
# k is the number of clusters
def kmeans1(X, k):
    m, n = X.shape
    if k > m:
        # logically, num clusters cannot excede observations
        raise ValueError(f"number of clusters {k} > number of observations {m}")

    # random initial assignment of each observation to a cluster
    clusters0 = np.random.randint(0, k, m)
    # holds the actual cluster assignments
    clusters = np.zeros(m)

    while True:

        # initialzes array of centroids
        c = np.zeros((k, n))

        # computes the cluster centroids
        for i in range(k):
            # cenroid of cluster i placed in c
            c[i, :] = centroid(X[clusters0 == i])

        # assign each observation to the nearest centroid
        for i in range(m):  # loop over each data point
            clusters[i] = 0  # initially assign to cluster 0
            best = distance(X[i, :], c[0, :])  # compute distance to first centroid

            # compare with other centroids
            for j in range(1, k):
                candidate = distance(X[i, :], c[j, :])  # compute distance

                if candidate < best:  # if closer, update assignment
                    best = candidate
                    clusters[i] = j

        # when the clusters stop changing
        if np.array_equal(clusters, clusters0):
            break

        clusters0 = clusters.copy()

    clusters_assigned = len(Counter(clusters))
    if clusters_assigned != k:
        raise ValueError(f"clustering solution contains {clusters_assigned} < {k} clusters.")
    return clusters


# computes the value of the kmeans objective
# function, the sum of the within-cluster distances.
# X is the matrix of observations.
# k is the number of clusters.
# cl is the clustering solution.
def objective(X, k, cl):
    sum_dist = 0  # tracks sum of the distances to centroids in each cluster
    for i in range(k):
        cluster_i_obs = X[cl == i]  # find all the points in X that are in cluster i
        if len(cluster_i_obs) > 0:  # finds centroid of the cluster only if there are points in it
            centroid_clust_i = centroid(cluster_i_obs)
            sum_dist += sum(
                distance(observation, centroid_clust_i) for observation in cluster_i_obs
            )  # adds this distance to the sum of distances for minimizing later
    return sum_dist


# driver function for kmeans.
# X is the (scaled) matrix of observations.
# k is the number of clusters.
# niter is the number of times to run the k-means algorithm.
# the best of the niter candidate solutions is returned.
def kmeans(X, k, niter=50):
    smallest_dist = float("inf")
    final_cl = np.zeros(X.shape[0])
    for i in range(niter):
        solution = kmeans1(X, k)
        sum_distances = objective(X, k, solution)
        if sum_distances < smallest_dist:
            smallest_dist = sum_distances
            final_cl = solution
    return final_cl


def main():
    # loads dataset
    US_Arrests_data = pd.read_csv("USArrests.csv")
    # extracts the state names
    US_states = US_Arrests_data['State']

    #
