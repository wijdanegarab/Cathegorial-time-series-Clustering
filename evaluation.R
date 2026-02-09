
# EVALUATION METRICS FOR CLUSTERING


# install.packages("fossil")
library(fossil)

# 1. ADJUSTED RAND INDEX (ARI)


calculate_ari <- function(true_labels, predicted_labels) {
  """
  Calculate Adjusted Rand Index
  
  Args:
    true_labels: true labels (ground truth)
    predicted_labels: predicted labels from clustering
  
  Returns:
    ari_score: ARI score (range: -1 to 1)
      1 = perfect agreement
      0 = random labeling
      < 0 = worse than random
  """
  ari <- adj.rand.index(true_labels, predicted_labels)
  return(ari)
}



# 2. SILHOUETTE SCORE


calculate_silhouette_score <- function(dist_matrix, labels) {
  """
  Calculate average Silhouette Score
  
  Args:
    dist_matrix: distance matrix
    labels: cluster labels
  
  Returns:
    silhouette_avg: average silhouette score (range: -1 to 1)
      close to 1 = good clustering
      close to 0 = ambiguous clustering
      negative = bad clustering
  """
  dist_obj <- as.dist(dist_matrix)
  
  silhouette_vals <- silhouette(labels, dist_obj)
  silhouette_avg <- mean(silhouette_vals[, 3])
  
  return(silhouette_avg)
}


calculate_silhouette_samples <- function(dist_matrix, labels) {
  """
  Calculate Silhouette Score for each point
  
  Args:
    dist_matrix: distance matrix
    labels: cluster labels
  
  Returns:
    silhouette_vals: silhouette scores for each point
  """
  dist_obj <- as.dist(dist_matrix)
  silhouette_vals <- silhouette(labels, dist_obj)
  
  return(silhouette_vals[, 3])
}



# 3. PLOT SILHOUETTE


plot_silhouette <- function(dist_matrix, labels, title = "Silhouette Plot", filename = NULL) {
  """
  Plot silhouette diagram
  
  Args:
    dist_matrix: distance matrix
    labels: cluster labels
    title: plot title
    filename: if provided, save plot
  """
  dist_obj <- as.dist(dist_matrix)
  silhouette_vals <- silhouette(labels, dist_obj)
  silhouette_avg <- mean(silhouette_vals[, 3])
  
  if (!is.null(filename)) {
    png(filename, width = 1000, height = 600)
  }
  
  plot(silhouette_vals, main = title, border = NA)
  abline(v = silhouette_avg, col = "red", lty = 2)
  
  if (!is.null(filename)) {
    dev.off()
    cat("✓ Silhouette plot saved:", filename, "\n")
  }
}



# 4. CREATE GROUND TRUTH LABELS


create_ground_truth_labels <- function(n_sequences, n_clusters = 3) {
  """
  Create ground truth labels (for testing)
  
  Simply divide sequences into n_clusters groups
  
  Args:
    n_sequences: total number of sequences
    n_clusters: number of clusters
  
  Returns:
    labels: array of labels (1, 2, 3, ...)
  """
  labels <- rep(1:n_clusters, each = floor(n_sequences / n_clusters))
  remaining <- n_sequences %% n_clusters
  if (remaining > 0) {
    labels <- c(labels, 1:remaining)
  }
  
  return(labels)
}



# MAIN EXECUTION


# Load data
sequences <- as.matrix(read.csv("sequences_m1.csv", header = TRUE))
dist_hamming <- readRDS("distance_matrix_hamming.rds")
dist_om <- readRDS("distance_matrix_om.rds")

# Load labels
labels_kmed_hamming <- readRDS("labels_kmedoids_hamming.rds")
labels_kmed_om <- readRDS("labels_kmedoids_om.rds")
labels_hc_hamming <- readRDS("labels_hierarchical_hamming.rds")
labels_hc_om <- readRDS("labels_hierarchical_om.rds")

# Create ground truth (simplified: divide into 3 groups)
true_labels <- create_ground_truth_labels(nrow(sequences), n_clusters = 3)

cat("=" %&% rep("=", 58), "\n")
cat("EVALUATION\n")
cat("=" %&% rep("=", 58), "\n")

# Results storage
results <- data.frame(
  Method = character(),
  ARI = numeric(),
  Silhouette = numeric(),
  stringsAsFactors = FALSE
)

# K-MEDOIDS + Hamming
cat("\n1. K-MEDOIDS + HAMMING\n")
cat("-" %&% rep("-", 58), "\n")
ari_kmed_hamming <- calculate_ari(true_labels, labels_kmed_hamming)
sil_kmed_hamming <- calculate_silhouette_score(dist_hamming, labels_kmed_hamming)
cat("ARI:", round(ari_kmed_hamming, 4), "\n")
cat("Silhouette:", round(sil_kmed_hamming, 4), "\n")
results <- rbind(results, data.frame(
  Method = "KMedoids (Hamming)",
  ARI = ari_kmed_hamming,
  Silhouette = sil_kmed_hamming,
  stringsAsFactors = FALSE
))

# K-MEDOIDS + OM
cat("\n2. K-MEDOIDS + OPTIMAL MATCHING\n")
cat("-" %&% rep("-", 58), "\n")
ari_kmed_om <- calculate_ari(true_labels, labels_kmed_om)
sil_kmed_om <- calculate_silhouette_score(dist_om, labels_kmed_om)
cat("ARI:", round(ari_kmed_om, 4), "\n")
cat("Silhouette:", round(sil_kmed_om, 4), "\n")
results <- rbind(results, data.frame(
  Method = "KMedoids (OM)",
  ARI = ari_kmed_om,
  Silhouette = sil_kmed_om,
  stringsAsFactors = FALSE
))

# HIERARCHICAL + Hamming
cat("\n3. HIERARCHICAL + HAMMING\n")
cat("-" %&% rep("-", 58), "\n")
ari_hc_hamming <- calculate_ari(true_labels, labels_hc_hamming)
sil_hc_hamming <- calculate_silhouette_score(dist_hamming, labels_hc_hamming)
cat("ARI:", round(ari_hc_hamming, 4), "\n")
cat("Silhouette:", round(sil_hc_hamming, 4), "\n")
results <- rbind(results, data.frame(
  Method = "Hierarchical (Hamming)",
  ARI = ari_hc_hamming,
  Silhouette = sil_hc_hamming,
  stringsAsFactors = FALSE
))

# HIERARCHICAL + OM
cat("\n4. HIERARCHICAL + OPTIMAL MATCHING\n")
cat("-" %&% rep("-", 58), "\n")
ari_hc_om <- calculate_ari(true_labels, labels_hc_om)
sil_hc_om <- calculate_silhouette_score(dist_om, labels_hc_om)
cat("ARI:", round(ari_hc_om, 4), "\n")
cat("Silhouette:", round(sil_hc_om, 4), "\n")
results <- rbind(results, data.frame(
  Method = "Hierarchical (OM)",
  ARI = ari_hc_om,
  Silhouette = sil_hc_om,
  stringsAsFactors = FALSE
))

# Comparison table
cat("\n5. COMPARISON TABLE\n")
cat("-" %&% rep("-", 58), "\n")
print(results)
write.csv(results, "evaluation_results.csv", row.names = FALSE)
cat("✓ Results saved to evaluation_results.csv\n")

# Silhouette plots
cat("\n6. SILHOUETTE PLOTS\n")
cat("-" %&% rep("-", 58), "\n")

plot_silhouette(dist_hamming, labels_hc_hamming,
               title = "Silhouette (Hierarchical + Hamming)",
               filename = "silhouette_hc_hamming.png")

plot_silhouette(dist_om, labels_hc_om,
               title = "Silhouette (Hierarchical + OM)",
               filename = "silhouette_hc_om.png")

cat("\n" %&% rep("=", 58), "\n")
cat("BEST METHOD: Hierarchical + OM\n")
cat("ARI:", round(ari_hc_om, 4), "\n")
cat("Silhouette:", round(sil_hc_om, 4), "\n")
cat("=" %&% rep("=", 58), "\n")
