Crime Data State Clustering
===========================

Project: K-Means Clustering of US Violent Crime Rates by State

This project applies K-Means clustering to the USArrests dataset to group U.S. states based on violent crime statistics. It then visualizes the clustering results on an interactive, color-coded map using Folium and a GeoJSON file of U.S. state boundaries.

The main workflow is implemented in:
    K_means_clustering_for_state_crimes.py

and uses the data file:
    USArrests.csv


--------------------------------------
Overview
--------------------------------------

The goal of this project is to:

1. Load violent crime statistics for each U.S. state (Murder, Assault, UrbanPop, Rape).
2. Standardize the numeric features for fair comparison.
3. Implement K-Means clustering "from scratch" (custom kmeans1 / kmeans / objective functions).
4. Assign each state to a cluster based on its standardized crime profile.
5. Compute actual (unscaled) mean crime rates per cluster.
6. Visualize the clusters on an interactive U.S. map using Folium, with:
   - Each state colored by cluster
   - A popup summary listing cluster sizes, mean crime rates, and member states

The code saves the interactive map as:
    us_clusters_map.html


--------------------------------------
Files
--------------------------------------

1. K_means_clustering_for_state_crimes.py
   - Python script that:
     - Loads and standardizes the USArrests data
     - Runs custom K-Means clustering
     - Computes cluster-wise mean crime rates
     - Builds and saves an interactive Folium map of clustered states as us_clusters_map.html

2. USArrests.csv
   - CSV file containing the crime data used for clustering, with columns:
     - State        : U.S. state name
     - Murder       : Murder arrests per 100
