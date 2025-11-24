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
    clusters0 = np.random.randint(0, k, m, dtype=int)
    # holds the actual cluster assignments
    clusters = np.zeros(m, dtype=int)

    while True:

        # initialzes array of centroids
        c = np.zeros((k, n))

        # computes the cluster centroids
        for i in range(k):
            # all points currently assigned to cluster i
            cluster_points = X[clusters0 == i]

            if cluster_points.size == 0:
                # if a cluster is empty, reinitialize its centroid
                # to a random data point to avoid NaNs
                rand_idx = np.random.randint(0, m)
                c[i, :] = X[rand_idx, :]
            else:
                # cenroid of cluster i placed in c
                c[i, :] = centroid(cluster_points)

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
    final_cl = None

    for i in range(niter):
        solution = kmeans1(X, k)
        clusters_assigned = len(np.unique(solution))

        # skip solutions that collapsed to fewer than k clusters
        if clusters_assigned < k:
            continue

        sum_distances = objective(X, k, solution)
        if sum_distances < smallest_dist:
            smallest_dist = sum_distances
            final_cl = solution

    if final_cl is None:
        raise ValueError(
            f"Could not find a solution with {k} non-empty clusters after {niter} runs."
        )

    return final_cl


def main():
    # loads dataset
    US_Arrests_data = pd.read_csv(
        r"C:\Users\angie\OneDrive\Documents\School\Arrests_Test\USArrests.csv"
    )
    # extracts the state names
    US_states = US_Arrests_data['State']

    # Extract numerical data before scaling to compute actual means later
    Original_State_Data = US_Arrests_data.iloc[:, 1:5].values

    # scales the data for clustering (scale each attribute to have mean 0 and std dev 1 )
    State_Data = Original_State_Data.copy()
    scaler = StandardScaler()
    # State_Data is now numpy array with standardized values
    State_Data = scaler.fit_transform(State_Data)

    # Run k-means
    num_clusters = 4
    cl = kmeans(State_Data, num_clusters)

    # Assign states to clusters
    states_in_cluster = defaultdict(list)
    for i, state in enumerate(US_states):
        states_in_cluster[int(cl[i])].append(state)

    # Compute actual means per cluster
    cluster_actual_means = {}
    for cluster, states in states_in_cluster.items():
        indices = [US_states[US_states == state].index[0] for state in states]
        cluster_actual_means[cluster] = np.mean(Original_State_Data[indices], axis=0)

    # U.S. state boundaries from GeoJSON
    geojson_url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    geojson_data = requests.get(geojson_url).json()

    # color generated for each cluster and each state mapped to its cluster color
    cluster_colors = {
        cluster: "#{:06x}".format(random.randint(0, 0xFFFFFF))
        for cluster in states_in_cluster
    }
    state_cluster_map = {
        state: cluster_colors[cluster]
        for cluster, states in states_in_cluster.items()
        for state in states
    }

    us_map = folium.Map(location=[37.8, -96], zoom_start=4)

    # styles each state based on its cluster
    def style_function(feature):
        state_name = feature['properties']['name']
        # default grey if state not found
        color = state_cluster_map.get(state_name, "#d3d3d3")
        return {'fillColor': color, 'color': 'black', 'weight': 1, 'fillOpacity': 0.7}

    # adds states layer with colors
    folium.GeoJson(
        geojson_data,
        name="Clusters",
        style_function=style_function
    ).add_to(us_map)

    # generates cluster summary HTML to be added to map
    cluster_summary_html = "<h4>Cluster Summary (Crime Rates per 100K)</h4><ul>"
    for cluster, states in states_in_cluster.items():
        means = cluster_actual_means[cluster]
        cluster_summary_html += (
            f"<li><b>Cluster {cluster + 1}</b>: "
            f"{len(states)} states<br>"
            f"Murder: {means[0]:.2f} per 100K<br>"
            f"Assault: {means[1]:.2f} per 100K<br>"
            f"UrbanPop: {means[2]:.2f}%<br>"
            f"Rape: {means[3]:.2f} per 100K<br>"
            f"States: {', '.join(states)}</li><br>"
        )
    cluster_summary_html += "</ul>"

    # adds floating text box with actual cluster means
    html_popup = folium.Html(cluster_summary_html, script=True)
    popup = folium.Popup(html_popup, max_width=600)
    folium.Marker(
        [49, -125],
        icon=folium.Icon(color="blue", icon="info-sign"),
        popup=popup
    ).add_to(us_map)

    # layer control
    folium.LayerControl().add_to(us_map)

    # saves and displays the map
    us_map.save("us_clusters_map.html")
    print("A new map with cluster means (per 100K) has been generated: us_clusters_map.html")


if __name__ == "__main__":
    main()
