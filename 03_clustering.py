import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def kmedoids_clustering(dist_matrix, n_clusters=3, max_iter=100, random_state=42):
    np.random.seed(random_state)
    n = dist_matrix.shape[0]
    medoids = np.random.choice(n, n_clusters, replace=False)
    
    for _ in range(max_iter):
        labels = np.argmin(dist_matrix[:, medoids], axis=1)
        old_cost = np.sum(np.min(dist_matrix[:, medoids], axis=1))
        
        improved = False
        for m_idx in range(n_clusters):
            for i in range(n):
                if i in medoids:
                    continue
                
                new_medoids = medoids.copy()
                new_medoids[m_idx] = i
                new_cost = np.sum(np.min(dist_matrix[:, new_medoids], axis=1))
                
                if new_cost < old_cost:
                    medoids = new_medoids
                    old_cost = new_cost
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    labels = np.argmin(dist_matrix[:, medoids], axis=1)
    return labels

def hierarchical_clustering(dist_matrix, n_clusters=3):
    condensed = squareform(dist_matrix)
    linkage_matrix = linkage(condensed, method='ward')
    labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
    return labels

dhd = np.load("distance_matrix_dhd.npy")
om = np.load("distance_matrix_om.npy")

labels_kmed_dhd = kmedoids_clustering(dhd)
labels_kmed_om = kmedoids_clustering(om)

labels_hc_dhd = hierarchical_clustering(dhd)
labels_hc_om = hierarchical_clustering(om)

np.save("labels_kmedoids_dhd.npy", labels_kmed_dhd)
np.save("labels_kmedoids_om.npy", labels_kmed_om)
np.save("labels_hierarchical_dhd.npy", labels_hc_dhd)
np.save("labels_hierarchical_om.npy", labels_hc_om)
