
# DISTANCE METRICS POUR SÉRIES TEMPORELLES CATÉGORIQUES



# 1. HAMMING DISTANCE


hamming_distance <- function(seq1, seq2) {
  """
  Calculate Hamming distance between two sequences
  
  Distance = number of positions where sequences differ
  
  Args:
    seq1, seq2: vectors of same length
  
  Returns:
    distance: number of different positions
  """
  if (length(seq1) != length(seq2)) {
    stop("Sequences must have the same length")
  }
  
  distance <- sum(seq1 != seq2)
  return(distance)
}


hamming_distance_matrix <- function(sequences) {
  """
  Create Hamming distance matrix for all pairs of sequences
  
  Args:
    sequences: matrix (n_sequences, seq_length)
  
  Returns:
    dist_matrix: symmetric matrix (n_sequences, n_sequences)
  """
  n_sequences <- nrow(sequences)
  dist_matrix <- matrix(0, nrow = n_sequences, ncol = n_sequences)
  
  for (i in 1:n_sequences) {
    for (j in i:n_sequences) {
      distance <- hamming_distance(sequences[i, ], sequences[j, ])
      dist_matrix[i, j] <- distance
      dist_matrix[j, i] <- distance
    }
  }
  
  return(dist_matrix)
}



# 2. OPTIMAL MATCHING DISTANCE (OM)


cost_matrix <- function(seq1, seq2, indel_cost = 1, substitution_cost = 1) {
  """
  Create cost matrix for Optimal Matching
  
  Args:
    seq1, seq2: sequences to compare
    indel_cost: cost of insertion/deletion
    substitution_cost: cost of substitution
  
  Returns:
    cost_matrix: matrix (len(seq1)+1, len(seq2)+1)
  """
  len1 <- length(seq1)
  len2 <- length(seq2)
  matrix_cost <- matrix(0, nrow = len1 + 1, ncol = len2 + 1)
  
  # Initialize first row and column (insertion costs)
  for (i in 1:(len1 + 1)) {
    matrix_cost[i, 1] <- (i - 1) * indel_cost
  }
  for (j in 1:(len2 + 1)) {
    matrix_cost[1, j] <- (j - 1) * indel_cost
  }
  
  # Fill matrix with dynamic programming
  for (i in 2:(len1 + 1)) {
    for (j in 2:(len2 + 1)) {
      # Cost if we substitute
      if (seq1[i-1] == seq2[j-1]) {
        substitution <- matrix_cost[i-1, j-1]  # No cost if identical
      } else {
        substitution <- matrix_cost[i-1, j-1] + substitution_cost
      }
      
      # Cost if we delete
      deletion <- matrix_cost[i-1, j] + indel_cost
      
      # Cost if we insert
      insertion <- matrix_cost[i, j-1] + indel_cost
      
      # Take minimum
      matrix_cost[i, j] <- min(substitution, deletion, insertion)
    }
  }
  
  return(matrix_cost)
}


optimal_matching_distance <- function(seq1, seq2, indel_cost = 1, substitution_cost = 1) {
  """
  Calculate Optimal Matching distance between two sequences
  
  This is the minimum cost to transform seq1 into seq2
  via insertions, deletions, substitutions
  
  Args:
    seq1, seq2: sequences to compare
    indel_cost: cost of insertion/deletion
    substitution_cost: cost of substitution
  
  Returns:
    distance: minimum cost
  """
  matrix_cost <- cost_matrix(seq1, seq2, indel_cost, substitution_cost)
  return(matrix_cost[length(seq1) + 1, length(seq2) + 1])
}


optimal_matching_matrix <- function(sequences, indel_cost = 1, substitution_cost = 1) {
  """
  Create Optimal Matching distance matrix for all pairs
  
  Args:
    sequences: matrix (n_sequences, seq_length)
    indel_cost: cost of insertion/deletion
    substitution_cost: cost of substitution
  
  Returns:
    dist_matrix: symmetric matrix (n_sequences, n_sequences)
  """
  n_sequences <- nrow(sequences)
  dist_matrix <- matrix(0, nrow = n_sequences, ncol = n_sequences)
  
  for (i in 1:n_sequences) {
    for (j in i:n_sequences) {
      distance <- optimal_matching_distance(
        sequences[i, ], 
        sequences[j, ], 
        indel_cost = indel_cost, 
        substitution_cost = substitution_cost
      )
      dist_matrix[i, j] <- distance
      dist_matrix[j, i] <- distance
    }
  }
  
  return(dist_matrix)
}



# MAIN EXECUTION


# Load sequences
sequences <- as.matrix(read.csv("sequences_m1.csv", header = TRUE))

cat("=" %&% rep("=", 58), "\n")
cat("CALCUL DES DISTANCE MATRICES\n")
cat("=" %&% rep("=", 58), "\n")

# Hamming Distance
cat("\n1. HAMMING DISTANCE\n")
cat("-" %&% rep("-", 58), "\n")
dist_hamming <- hamming_distance_matrix(sequences)
cat("Hamming distance matrix shape:", dim(dist_hamming), "\n")
non_zero_hamming <- dist_hamming[dist_hamming > 0]
cat("Min distance:", min(non_zero_hamming), "\n")
cat("Max distance:", max(dist_hamming), "\n")
cat("Mean distance:", mean(non_zero_hamming), "\n")

# Optimal Matching
cat("\n2. OPTIMAL MATCHING DISTANCE\n")
cat("-" %&% rep("-", 58), "\n")
dist_om <- optimal_matching_matrix(sequences)
cat("Optimal Matching distance matrix shape:", dim(dist_om), "\n")
non_zero_om <- dist_om[dist_om > 0]
cat("Min distance:", min(non_zero_om), "\n")
cat("Max distance:", max(dist_om), "\n")
cat("Mean distance:", mean(non_zero_om), "\n")

# Save
cat("\n3. SAUVEGARDE\n")
cat("-" %&% rep("-", 58), "\n")
saveRDS(dist_hamming, "distance_matrix_hamming.rds")
saveRDS(dist_om, "distance_matrix_om.rds")
cat("✓ Distance matrices saved\n")
