# Duplicate Detection Using LSH and Complete Linkage Clustering

This project implements a product matching system that identifies duplicate or similar products in a dataset of product listings. The system leverages MinHashing and Locality-Sensitive Hashing (LSH) for efficient duplicate detection using product titles.

The goal is to match products that are likely duplicates or variants by analyzing the similarities in their titles while avoiding expensive pairwise comparisons for large datasets.

## Code Structure
- Data Loading and Cleaning:
  Loads product data from a JSON file and cleans the product titles, removing irrelevant characters and standardizing text.

- Feature Extraction and create binary matrix:
  Extracts model words from product titles and uses those model words to construct a binary matrix of which titles contains which model words.

- MinHashing:
  Generates signatures for product titles using MinHashing, which compresses the product features into compact representations.

- Locality-Sensitive Hashing (LSH):
  Uses LSH to group similar products together, reducing the number of comparisons needed.

- Create dissimilarity matrix:
  A dissimilarity matrix is constructed based on the jaccard dissimilarity
  
-Clustering:
  Applies complete linkage clustering to the dissimilarity matrix to identify duplicate products.

-Evaluation:
  Measures the performance of the algorithm using precision, recall, F1 score, and F1* score, for different number of bands and bootstrap iterations.

-Visualization:
  Plots graphs to visualize the relationship between the fraction of comparisons and the evaluation metrics.

## How to use the code
First, make sure to install the required libraries before running the code. Then you import the data, clean the data, extract the model words and create a binary matrix with the associated functions. Then you initialize the number of hashes,  different numbers of bands and the number of bootstraps that you want to use. The bootstrap function is used to run the min-hashing, lsh, clustering and to construct a dataframe with all the metrics(pair quality, pair completeness, F1 and F1*) for each bootstrap and number of bands. Finally, we extract the metrics from the dataframe to construct the plots that we need.

