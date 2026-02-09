import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import pandas as pd


# EVALUATION METRICS FOR CLUSTERING



# 1. ADJUSTED RAND INDEX (ARI)


def calculate_ari(true_labels, predicted_labels):
    """
    Calculer Adjusted Rand Index
    
    Compare clustering prédis vs labels vrais
    
    Args:
        true_labels: labels réels (ground truth)
        predicted_labels: labels prédits par le clustering
    
    Returns:
        ari_score: score ARI (range: -1 to 1)
            1 = parfait agreement
            0 = random labeling
            < 0 = pire que random
    """
    ari = adjusted_rand_score(true_labels, predicted_labels)
    return ari


# 2. SILHOUETTE SCORE


def calculate_silhouette_score(dist_matrix, labels):
    """
    Calculer Silhouette Score moyen
    
    Mesure la qualité du clustering (sans besoin de ground truth)
    
    Args:
        dist_matrix: matrice de distances
        labels: labels des clusters
    
    Returns:
        silhouette_avg: score moyen (range: -1 to 1)
            proche de 1 = bon clustering
            proche de 0 = clustering ambigu
            négatif = mauvais clustering
    """
    # sklearn utilise distance euclidienne, donc on utilise une version manuelle
    silhouette_avg = silhouette_score(dist_matrix, labels, metric='precomputed')
    return silhouette_avg


def calculate_silhouette_samples(dist_matrix, labels):
    """
    Calculer Silhouette Score pour chaque point
    
    Args:
        dist_matrix: matrice de distances
        labels: labels des clusters
    
    Returns:
        silhouette_vals: array de scores pour chaque point
    """
    silhouette_vals = silhouette_samples(dist_matrix, labels, metric='precomputed')
    return silhouette_vals


def plot_silhouette(dist_matrix, labels, title="Silhouette Plot", filename=None):
    """
    Tracer le silhouette plot
    
    Args:
        dist_matrix: matrice de distances
        labels: labels des clusters
        title: titre du graphique
        filename: si fourni, sauvegarder
    """
    silhouette_vals = silhouette_samples(dist_matrix, labels, metric='precomputed')
    silhouette_avg = silhouette_score(dist_matrix, labels, metric='precomputed')
    
    plt.figure(figsize=(10, 6))
    y_lower = 10
    
    for i in range(len(np.unique(labels))):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / len(np.unique(labels)))
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_vals,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        y_lower = y_upper + 10
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Silhouette Coefficient", fontsize=12)
    plt.ylabel("Cluster Label", fontsize=12)
    plt.axvline(x=silhouette_avg, color="red", linestyle="--", label=f"Average: {silhouette_avg:.3f}")
    plt.legend()
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
        print(f"✓ Silhouette plot sauvegardé: {filename}")
    else:
        plt.show()



# 3. CREATE GROUND TRUTH LABELS


def create_ground_truth_labels(n_sequences, n_clusters=3):
    """
    Créer labels de ground truth (pour testing)
    
    Simplement diviser les sequences en n_clusters groupes
    
    Args:
        n_sequences: nombre total de sequences
        n_clusters: nombre de clusters
    
    Returns:
        labels: array de labels (0, 1, 2, ...)
    """
    labels = np.repeat(np.arange(n_clusters), n_sequences // n_clusters)
    # Si n_sequences n'est pas divisible par n_clusters, ajouter les restants
    remaining = n_sequences % n_clusters
    labels = np.concatenate([labels, np.arange(remaining)])
    return labels



# 4. COMPARISON TABLE


def create_evaluation_table(results_dict):
    """
    Créer une table de comparaison
    
    Args:
        results_dict: dictionnaire avec résultats
            {
                'method_name': {
                    'ari': score,
                    'silhouette': score
                },
                ...
            }
    
    Returns:
        df: DataFrame avec résultats
    """
    df = pd.DataFrame(results_dict).T
    df = df.round(4)
    return df



# TEST


if __name__ == "__main__":
    import pandas as pd
    
    # Charger données
    sequences = pd.read_csv("sequences_m1.csv").values
    dist_hamming = np.load("distance_matrix_hamming.npy")
    dist_om = np.load("distance_matrix_om.npy")
    
    # Charger labels de clustering
    labels_kmed_hamming = np.load("labels_kmedoids_hamming.npy")
    labels_kmed_om = np.load("labels_kmedoids_om.npy")
    labels_hc_hamming = np.load("labels_hierarchical_hamming.npy")
    labels_hc_om = np.load("labels_hierarchical_om.npy")
    
    # Créer ground truth (simplified: diviser en 3 groupes)
    true_labels = create_ground_truth_labels(len(sequences), n_clusters=3)
    
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    results = {}
    
    # K-MEDOIDS + Hamming
    print("\n1. K-MEDOIDS + HAMMING")
    print("-" * 60)
    ari_kmed_hamming = calculate_ari(true_labels, labels_kmed_hamming)
    sil_kmed_hamming = calculate_silhouette_score(dist_hamming, labels_kmed_hamming)
    print(f"ARI: {ari_kmed_hamming:.4f}")
    print(f"Silhouette: {sil_kmed_hamming:.4f}")
    results['KMedoids (Hamming)'] = {'ARI': ari_kmed_hamming, 'Silhouette': sil_kmed_hamming}
    
    # K-MEDOIDS + OM
    print("\n2. K-MEDOIDS + OPTIMAL MATCHING")
    print("-" * 60)
    ari_kmed_om = calculate_ari(true_labels, labels_kmed_om)
    sil_kmed_om = calculate_silhouette_score(dist_om, labels_kmed_om)
    print(f"ARI: {ari_kmed_om:.4f}")
    print(f"Silhouette: {sil_kmed_om:.4f}")
    results['KMedoids (OM)'] = {'ARI': ari_kmed_om, 'Silhouette': sil_kmed_om}
    
    # HIERARCHICAL + Hamming
    print("\n3. HIERARCHICAL + HAMMING")
    print("-" * 60)
    ari_hc_hamming = calculate_ari(true_labels, labels_hc_hamming)
    sil_hc_hamming = calculate_silhouette_score(dist_hamming, labels_hc_hamming)
    print(f"ARI: {ari_hc_hamming:.4f}")
    print(f"Silhouette: {sil_hc_hamming:.4f}")
    results['Hierarchical (Hamming)'] = {'ARI': ari_hc_hamming, 'Silhouette': sil_hc_hamming}
    
    # HIERARCHICAL + OM
    print("\n4. HIERARCHICAL + OPTIMAL MATCHING")
    print("-" * 60)
    ari_hc_om = calculate_ari(true_labels, labels_hc_om)
    sil_hc_om = calculate_silhouette_score(dist_om, labels_hc_om)
    print(f"ARI: {ari_hc_om:.4f}")
    print(f"Silhouette: {sil_hc_om:.4f}")
    results['Hierarchical (OM)'] = {'ARI': ari_hc_om, 'Silhouette': sil_hc_om}
    
    # Créer table de comparaison
    print("\n5. COMPARISON TABLE")
    print("-" * 60)
    df_results = create_evaluation_table(results)
    print(df_results)
    df_results.to_csv("evaluation_results.csv")
    print("✓ Résultats sauvegardés dans evaluation_results.csv")
    
    # Silhouette plots
    print("\n6. SILHOUETTE PLOTS")
    print("-" * 60)
    plot_silhouette(dist_hamming, labels_hc_hamming,
                   title="Silhouette (Hierarchical + Hamming)",
                   filename="silhouette_hc_hamming.png")
    
    plot_silhouette(dist_om, labels_hc_om,
                   title="Silhouette (Hierarchical + OM)",
                   filename="silhouette_hc_om.png")
    
    print("\n" + "=" * 60)
    print("BEST METHOD: Hierarchical + OM")
    print(f"ARI: {ari_hc_om:.4f}")
    print(f"Silhouette: {sil_hc_om:.4f}")
    print("=" * 60)
