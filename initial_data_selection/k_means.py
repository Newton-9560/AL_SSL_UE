from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

def split_by_kmeans_with_pca_multiple(data_list, n_clusters=100, n_select=300, pca_dim=128):
    """
    Cluster data with PCA + KMeans and select n_select samples proportionally from clusters.

    Parameters:
        data_list (list of dict): Input dataset, each dict must contain 'hidden_state'.
        n_clusters (int): Number of clusters for KMeans.
        n_select (int): Total number of samples to select.
        pca_dim (int): Number of PCA dimensions before KMeans.

    Returns:
        selected_data (list of dict): Selected representative subset.
        remaining_data (list of dict): The rest of the dataset.
    """
    vectors = np.array([item['hidden_state'] for item in data_list])

    pca = PCA(n_components=pca_dim, random_state=42)
    reduced_vectors = pca.fit_transform(vectors)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(reduced_vectors)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    samples_per_cluster = [n_select // n_clusters] * n_clusters
    for i in range(n_select % n_clusters):
        samples_per_cluster[i] += 1
    # (distribute the remainder evenly)

    selected_indices = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue  # Skip empty cluster

        num_to_select = min(samples_per_cluster[cluster_id], len(cluster_indices))

        cluster_vectors = reduced_vectors[cluster_indices]
        center = centers[cluster_id]

        distances = np.linalg.norm(cluster_vectors - center, axis=1)

        closest_indices_in_cluster = cluster_indices[np.argsort(distances)[:num_to_select]]

        selected_indices.extend(closest_indices_in_cluster)

    selected_set = set(selected_indices)
    selected_data = [data_list[i] for i in selected_indices]
    remaining_data = [item for i, item in enumerate(data_list) if i not in selected_set]

    return selected_data, remaining_data


def split_by_kmeans_multiple_without_pca(data_list, n_clusters=100, n_select=300):
    """
    Cluster raw hidden_state vectors with KMeans (no PCA) and select n_select samples,
    roughly evenly distributed across clusters.

    Parameters:
        data_list (list of dict): Dataset with 'hidden_state' in each dict.
        n_clusters (int): Number of KMeans clusters.
        n_select (int): Total number of samples to select.

    Returns:
        selected_data (list of dict): Selected representative subset.
        remaining_data (list of dict): Remaining data.
    """
    vectors = np.array([item['hidden_state'] for item in data_list])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(vectors)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    samples_per_cluster = [n_select // n_clusters] * n_clusters
    for i in range(n_select % n_clusters):
        samples_per_cluster[i] += 1

    selected_indices = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue  # Skip empty cluster

        num_to_select = min(samples_per_cluster[cluster_id], len(cluster_indices))

        cluster_vectors = vectors[cluster_indices]
        center = centers[cluster_id]

        distances = np.linalg.norm(cluster_vectors - center, axis=1)
        closest_indices = cluster_indices[np.argsort(distances)[:num_to_select]]

        selected_indices.extend(closest_indices)

    selected_set = set(selected_indices)
    selected_data = [data_list[i] for i in selected_indices]
    remaining_data = [item for i, item in enumerate(data_list) if i not in selected_set]

    return selected_data, remaining_data
