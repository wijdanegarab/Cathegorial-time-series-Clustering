import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
import pandas as pd

def calculate_ari(true_labels, predicted_labels):
    return adjusted_rand_score(true_labels, predicted_labels)

def calculate_silhouette(dist_matrix, labels):
    return silhouette_score(dist_matrix, labels, metric='precomputed')

def create_ground_truth_labels(n_sequences, n_clusters=3):
    labels = np.repeat(np.arange(n_clusters), n_sequences // n_clusters)
    remaining = n_sequences % n_clusters
    labels = np.concatenate([labels, np.arange(remaining)])
    return labels

dhd = np.load("distance_matrix_dhd.npy")
om = np.load("distance_matrix_om.npy")

labels_kmed_dhd = np.load("labels_kmedoids_dhd.npy")
labels_kmed_om = np.load("labels_kmedoids_om.npy")
labels_hc_dhd = np.load("labels_hierarchical_dhd.npy")
labels_hc_om = np.load("labels_hierarchical_om.npy")

true_labels = create_ground_truth_labels(150, n_clusters=3)

results = {
    'KMedoids (DHD)': {
        'ARI': calculate_ari(true_labels, labels_kmed_dhd),
        'ASW': calculate_silhouette(dhd, labels_kmed_dhd)
    },
    'KMedoids (OM)': {
        'ARI': calculate_ari(true_labels, labels_kmed_om),
        'ASW': calculate_silhouette(om, labels_kmed_om)
    },
    'Hierarchical (DHD)': {
        'ARI': calculate_ari(true_labels, labels_hc_dhd),
        'ASW': calculate_silhouette(dhd, labels_hc_dhd)
    },
    'Hierarchical (OM)': {
        'ARI': calculate_ari(true_labels, labels_hc_om),
        'ASW': calculate_silhouette(om, labels_hc_om)
    }
}

df_results = pd.DataFrame(results).T.round(4)
df_results.to_csv("evaluation_results.csv")
