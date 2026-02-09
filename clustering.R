
# CLUSTERING ALGORITHMS


# Install/Load packages if needed
# install.packages("cluster")
# install.packages("dendextend")

library(cluster)


# 1. K-MEDOIDS CLUSTERING


kmedoids_clustering <- function(dist_matrix, n_clusters = 3) {
  """
  Apply K-Medoids clustering using PAM algorithm
  
  Args:
    dist_matrix: distance matrix (n_samples, n_samples)
    n_clusters: number of clusters
  
  Returns:
    list with:
      labels: cluster labels (1, 2, 3, ...)
      medoids: indices of medoids
  """
  # Convert to dist object
  dist_obj <- as.dist(dist_matrix)
  
  # Apply PAM (Partitioning Around Medoids)
  pam_result <- pam(dist_obj, k = n_clusters, diss = TRUE)
  
  return(list(
    labels = pam_result$clustering,
    medoids = pam_result$medoids
  ))
}



# 2. HIERARCHICAL CLUSTERING


hierarchical_clustering <- function(dist_matrix, n_clusters = 3, linkage_method = "ward") {
  """
  Apply Hierarchical Clustering (Agglomerative)
  
  Args:
    dist_matrix: distance matrix (n_samples, n_samples)
    n_clusters: number of clusters
    linkage_method: 'ward', 'complete', 'average', 'single'
  
  Returns:
    list with:
      labels: cluster labels (1, 2, 3, ...)
      hclust_obj: hclust object (for dendrogram)
  """
  # Convert to dist object
  dist_obj <- as.dist(dist_matrix)
  
  # Hierarchical clustering
  hc <- hclust(dist_obj, method = linkage_method)
  
  # Cut dendrogram at n_clusters
  labels <- cutree(hc, k = n_clusters)
  
  return(list(
    labels = labels,
    hclust_obj = hc
  ))
}



# 3. PLOT DENDROGRAM


plot_dendrogram <- function(hclust_obj, title = "Dendrogram", filename = NULL) {
  """
  Plot dendrogram
  
  Args:
    hclust_obj: hclust object
    title: title of plot
    filename: if provided, save plot
  """
  if (!is.null(filename)) {
    png(filename, width = 1200, height = 600)
  }
  
  plot(hclust_obj, main = title, xlab = "Sample Index", ylab = "Distance")
  
  if (!is.null(filename)) {
    dev.off()
    cat("✓ Dendrogram saved:", filename, "\n")
  }
}



# 4. PLOT CLUSTERS (2D using MDS)


plot_clusters_2d <- function(dist_matrix, labels, title = "Clusters", filename = NULL) {
  """
  Plot clusters in 2D using MDS
  
  Args:
    dist_matrix: distance matrix
    labels: cluster labels
    title: plot title
    filename: if provided, save plot
  """
  # MDS for dimensionality reduction
  mds_result <- cmdscale(as.dist(dist_matrix), k = 2)
  
  if (!is.null(filename)) {
    png(filename, width = 1000, height = 800)
  }
  
  colors <- c("red", "blue", "green", "purple", "orange", "brown")
  unique_clusters <- unique(labels)
  
  plot(mds_result, main = title, xlab = "MDS Dimension 1", ylab = "MDS Dimension 2",
       col = colors[labels], pch = 19, cex = 1.5)
  
  legend("topright", legend = paste("Cluster", unique_clusters),
         col = colors[unique_clusters], pch = 19)
  
  if (!is.null(filename)) {
    dev.off()
    cat("✓ Plot saved:", filename, "\n")
  }
}



# MAIN EXECUTION


# Load data
sequences <- as.matrix(read.csv("sequences_m1.csv", header = TRUE))
dist_hamming <- readRDS("distance_matrix_hamming.rds")
dist_om <- readRDS("distance_matrix_om.rds")

cat("=" %&% rep("=", 58), "\n")
cat("CLUSTERING\n")
cat("=" %&% rep("=", 58), "\n")

# K-MEDOIDS + Hamming
cat("\n1. K-MEDOIDS + HAMMING DISTANCE\n")
cat("-" %&% rep("-", 58), "\n")
kmed_hamming <- kmedoids_clustering(dist_hamming, n_clusters = 3)
cat("Labels:", unique(kmed_hamming$labels), "\n")
cat("Cluster sizes:", table(kmed_hamming$labels), "\n")
cat("Medoids:", kmed_hamming$medoids, "\n")

# K-MEDOIDS + OM
cat("\n2. K-MEDOIDS + OPTIMAL MATCHING\n")
cat("-" %&% rep("-", 58), "\n")
kmed_om <- kmedoids_clustering(dist_om, n_clusters = 3)
cat("Labels:", unique(kmed_om$labels), "\n")
cat("Cluster sizes:", table(kmed_om$labels), "\n")
cat("Medoids:", kmed_om$medoids, "\n")

# HIERARCHICAL + Hamming
cat("\n3. HIERARCHICAL + HAMMING DISTANCE\n")
cat("-" %&% rep("-", 58), "\n")
hc_hamming <- hierarchical_clustering(dist_hamming, n_clusters = 3, linkage_method = "ward")
cat("Labels:", unique(hc_hamming$labels), "\n")
cat("Cluster sizes:", table(hc_hamming$labels), "\n")

# HIERARCHICAL + OM
cat("\n4. HIERARCHICAL + OPTIMAL MATCHING\n")
cat("-" %&% rep("-", 58), "\n")
hc_om <- hierarchical_clustering(dist_om, n_clusters = 3, linkage_method = "ward")
cat("Labels:", unique(hc_om$labels), "\n")
cat("Cluster sizes:", table(hc_om$labels), "\n")

# VISUALIZATIONS
cat("\n5. VISUALIZATIONS\n")
cat("-" %&% rep("-", 58), "\n")

plot_dendrogram(hc_hamming$hclust_obj,
               title = "Hierarchical Clustering Dendrogram (Hamming)",
               filename = "dendrogram_hamming.png")

plot_dendrogram(hc_om$hclust_obj,
               title = "Hierarchical Clustering Dendrogram (OM)",
               filename = "dendrogram_om.png")

plot_clusters_2d(dist_hamming, hc_hamming$labels,
                title = "HC Clusters (Hamming)",
                filename = "clusters_hc_hamming.png")

plot_clusters_2d(dist_om, hc_om$labels,
                title = "HC Clusters (OM)",
                filename = "clusters_hc_om.png")

# Save labels
saveRDS(kmed_hamming$labels, "labels_kmedoids_hamming.rds")
saveRDS(kmed_om$labels, "labels_kmedoids_om.rds")
saveRDS(hc_hamming$labels, "labels_hierarchical_hamming.rds")
saveRDS(hc_om$labels, "labels_hierarchical_om.rds")

cat("✓ Labels saved\n")
