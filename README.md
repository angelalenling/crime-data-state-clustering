Crime Data State Clustering
===========================

Project: K-Means Clustering of US Violent Crime Rates by State

This project applies K-Means clustering to the USArrests dataset to group U.S. states based on violent crime statistics. It then visualizes the clustering results on an interactive, color-coded map using Folium and a GeoJSON file of U.S. state boundaries.

The main workflow is implemented in:
    K_means_clustering_for_state_crimes.ipynb

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

The script saves the interactive map as:
    us_clusters_map.html


--------------------------------------
Files
--------------------------------------

K_means_clustering_for_state_crimes.ipynb
    Jupyter Notebook that:
    - Loads and standardizes the USArrests data
    - Runs custom K-Means clustering
    - Computes cluster-wise mean crime rates
    - Builds and saves an interactive Folium map of clustered states

USArrests.csv
    CSV file containing the crime data used for clustering, with columns:
    - State        : U.S. state name
    - Murder       : Murder arrests per 100,000 residents
    - Assault      : Assault arrests per 100,000 residents
    - UrbanPop     : Percent urban population
    - Rape         : Rape arrests per 100,000 residents


--------------------------------------
Methodology
--------------------------------------

1. Data Loading
   - Read USArrests.csv into a pandas DataFrame.
   - Extract state names and numeric crime features.

2. Standardization
   - Use StandardScaler to transform numeric columns so that each has mean 0 and standard deviation 1.
   - Keep a copy of the original (unscaled) data to compute actual crime rate means per cluster later.

3. Custom K-Means Implementation
   - distance(x, y): squared Euclidean distance between two vectors.
   - centroid(X): mean of all points in a cluster.
   - kmeans1(X, k):
       * Randomly assigns each observation to an initial cluster.
       * Iteratively:
         - Computes centroids of each cluster.
         - Reassigns each point to its nearest centroid.
       * Stops when cluster assignments no longer change.
       * Checks that all k clusters are non-empty.
   - objective(X, k, cl):
       * Computes the sum of within-cluster distances to measure cluster compactness.
   - kmeans(X, k, niter=50):
       * Runs kmeans1 multiple times with different random initializations.
       * Keeps the solution with the smallest objective value.

4. Clustering
   - Choose a number of clusters (e.g., k = 4).
   - Run kmeans(...) on the standardized State_Data.
   - Group states by their assigned cluster.
   - For each cluster, compute the mean of the original (unscaled) crime variables (Murder, Assault, UrbanPop, Rape).

5. Mapping with Folium
   - Load U.S. state boundaries from a public GeoJSON file.
   - Assign a distinct random color to each cluster.
   - Build a Folium map centered on the U.S.
   - Use a style function to color each state according to its cluster.
   - Generate an HTML summary listing, for each cluster:
       * Number of states
       * Mean crime rates per 100,000 (and UrbanPop %)
       * List of states in that cluster
   - Add this summary as a popup marker on the map.
   - Save as:
       us_clusters_map.html


--------------------------------------
How To Run
--------------------------------------

1. Install required packages:

    pip install pandas numpy folium requests scikit-learn

2. Ensure the following files are in the same project folder:

    K_means_clustering_for_state_crimes.ipynb
    USArrests.csv

3. Open the notebook in Jupyter (or VS Code, etc.) and run all cells.
   After execution, you should see a message:

    A new map with cluster means (per 100K) has been generated: us_clusters_map.html

4. Open us_clusters_map.html in a web browser to explore the interactive map.


--------------------------------------
Data Source and Attribution
--------------------------------------

The data used in USArrests.csv is based on the classic USArrests dataset from the R "datasets" package. It contains statistics on violent crime in the United States for 1973:

- Murder, Assault, and Rape: arrests per 100,000 residents.
- UrbanPop: percentage of the population living in urban areas.

Original data are documented as "Violent Crime Rates by US State, 1973" and are widely distributed as part of the R base datasets.

This repository uses a CSV version of that dataset for educational and non-commercial purposes only.
