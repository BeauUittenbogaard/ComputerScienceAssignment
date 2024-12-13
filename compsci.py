import numpy as np
import pandas as pd
import json
import re
import itertools
from sklearn.metrics import f1_score
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import matplotlib.pyplot as plt

# Load the JSON data from the file
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Extract titles, shops, and brands from the data
def titles(data):
    titles = []
    for key, value in data.items():
        for item in value:
            titles.append((key, item.get('title'), item.get('shop'), item.get('featuresMap').get('Brand')))
    return titles


# Clean the title text by converting to lowercase and removing unnecessary characters
def data_cleaning(data):
    data['title'] = data['title'].str.lower()
    data['brand'] = data['brand'].str.lower()
    data['shop'] = data['shop'].str.lower()

    # Replace variations of 'inch' and 'hz' to standardize
    variations_inch = ['inches', '"', '-inch', ' inch', '”', "'", '\"']
    for variation in variations_inch:
        data['title'] = data['title'].str.replace(variation, "inch")

    variations_hz = ['hertz', '-hz', ' hz']
    for variation in variations_hz:
        data['title'] = data['title'].str.replace(variation, "hz")

    # Remove website-related words from titles
    website_words = ['amazon', 'amazon.com', 'newegg', 'newegg.com', 'best-buy', 'best buy', 'best-buy.com', 'thenerds', 'thenerds.net', 'nerds', '.com']
    for words in website_words:
        data['title'] = data['title'].str.replace(words, '')

    # Remove special characters
    chars = "`~!@#$%^&*()-_+;<>?|[]'"
    for char in chars:
        data['title'] = data['title'].str.replace(char, '')

    data['title'] = data['title'].str.replace(" / ", " ")
    return data


# Extract model words (numbers and letters) from the product titles
def extract_model_words(title):
    regex = re.compile(r'([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)')
    model_words = regex.findall(title)
    return model_words


# Extract all unique model words from the dataset
def extract_all_model_words(data):
    all_model_words = set()
    for title in data['title']:
        model_words = extract_model_words(title)
        all_model_words.update([word for group in model_words for word in group if word])

    return sorted(all_model_words)


# Create a binary matrix where each row corresponds to a model word and each column corresponds to a product
def create_binary_vector(model_words, titles):
    binary_matrix = np.zeros((len(model_words), len(titles)), dtype=int)

    for i, word in enumerate(model_words):
        for j, title in enumerate(titles):
            if word in title:
                binary_matrix[i, j] = 1

    return binary_matrix


# Find the next prime number greater than or equal to n (used for hashing)
def next_prime_number(n):
    def prime_number(number):
        if number <= 1:
            return False
        for i in range(2, int(np.sqrt(number)) + 1):
            if number % i == 0:
                return False
        return True

    while not prime_number(n):
        n += 1
    return n


# Perform min hashing on the binary matrix to generate a signature matrix
def min_hashing(binary_matrix, number_of_hashes):
    number_of_model_words, number_of_products = binary_matrix.shape
    signature_matrix = np.full((number_of_hashes, number_of_products), np.inf)

    # Generate random hash functions using permutations
    np.random.seed(42)  # Ensures reproducibility
    a = np.random.randint(1, number_of_model_words, size=number_of_hashes)
    b = np.random.randint(0, number_of_model_words, size=number_of_hashes)
    prime = next_prime_number(number_of_model_words)

    for i in range(number_of_hashes):
        for v in range(number_of_products):
            # Compute hash values for rows with 1s
            hash_values = [(a[i] * row + b[i]) % prime for row in range(number_of_model_words) if binary_matrix[row, v] == 1]

            if hash_values:  # Ensure at least one value exists
                signature_matrix[i, v] = min(hash_values)

    return signature_matrix


# Perform Locality Sensitive Hashing (LSH) to find candidate pairs for potential duplicates
def perform_lsh(signature_matrix, number_of_bands, rows_per_band):
    number_of_hashes, number_of_products = signature_matrix.shape
    assert number_of_hashes == number_of_bands * rows_per_band, "number_of_hashes must equal number_of_bands * rows_per_band"

    # Split the signature matrix into bands
    buckets = defaultdict(list)
    for band_idx in range(number_of_bands):
        start_row = band_idx * rows_per_band
        end_row = start_row + rows_per_band

        band = signature_matrix[start_row:end_row, :]
        for v in range(number_of_products):
            # Use tuple of band rows + band index to define bucket
            band_id = tuple(band[:, v]) + (band_idx,)
            buckets[band_id].append(v)

    return buckets


# Find product pairs within the same bucket (candidate pairs)
def find_pairs_in_bucket(buckets):
    LSH_pairs = set()
    for items_in_bucket in buckets.values():
        if len(items_in_bucket) > 1:
            for pair in itertools.combinations(items_in_bucket, 2):
                LSH_pairs.add(tuple(sorted(pair)))

    return LSH_pairs


# Calculate the Jaccard similarity between two sets
def jaccard(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


# Build a dissimilarity matrix based on the Jaccard similarity of candidate pairs
def dissimilarity_matrix(lsh_pairs, binary_matrix, data):
    number_of_products = binary_matrix.shape[1]
    dissimilarity_matrix = np.ones((number_of_products, number_of_products)) * 1000

    for pair in lsh_pairs:
        product1, product2 = pair

        # Skip pairs from the same shop or brand
        if data['shop'][product1] == data['shop'][product2]:
            dissimilarity_matrix[product1, product2] = 1000
            dissimilarity_matrix[product2, product1] = 1000
            continue

        brand1, brand2 = data['brand'][product1], data['brand'][product2]
        if brand1 is not None and brand2 is not None and brand1 != brand2:
            dissimilarity_matrix[product1, product2] = 1000
            dissimilarity_matrix[product2, product1] = 1000
            continue

        set1 = set(np.where(binary_matrix[:, product1] == 1)[0])
        set2 = set(np.where(binary_matrix[:, product2] == 1)[0])

        distance = 1 - jaccard(set1, set2)
        dissimilarity_matrix[product1, product2] = distance
        dissimilarity_matrix[product2, product1] = distance

    np.fill_diagonal(dissimilarity_matrix, 1000)

    return dissimilarity_matrix


# Perform complete linkage clustering on the dissimilarity matrix
def cluster_products(dissimilarity_matrix, distance_threshold):
    clustering = AgglomerativeClustering(n_clusters= None, linkage='complete', distance_threshold=distance_threshold)
    cluster_labels = clustering.fit_predict(dissimilarity_matrix)
    cluster_dict = defaultdict(list)

    for idx, label in enumerate(cluster_labels):
        cluster_dict[label].append(idx)

    duplicate_pairs = set()
    for indices in cluster_dict.values():
        if len(indices) > 1:
            duplicate_pairs.update(itertools.combinations(indices, 2))

    return duplicate_pairs, dict(cluster_dict)


# Compute recall and precision from true and predicted pairs
def recall_and_precision(true_pairs, predicted_pairs):
    true_positives = len(predicted_pairs.intersection(true_pairs))
    recall = true_positives / len(true_pairs) if len(true_pairs) > 0  else 0
    precision = true_positives / len(predicted_pairs) if len(predicted_pairs) > 0 else 0
    return recall, precision


# Calculate the F1 score from recall and precision
def f1_score(recall, precision):
    return (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0


# Evaluate the scalability and quality of the clustering results
def evaluate_scalability(candidate_pairs, true_pairs, data):
    n_candidate = len(candidate_pairs)
    true_positives = len(candidate_pairs.intersection(true_pairs))
    n_true = len(true_pairs)
    N = len(data)
    total_number_of_comparisons = (N * (N - 1)) / 2

    pair_quality = true_positives / n_candidate if n_candidate > 0 else 0
    pair_completeness = true_positives / n_true if n_true > 0 else 0

    f1_star = (2 * pair_quality * pair_completeness) / (pair_quality + pair_completeness) if (pair_quality + pair_completeness) > 0 else 0
    fraction = n_candidate / total_number_of_comparisons

    return [fraction, pair_quality, pair_completeness, f1_star]


# Generate the true duplicates from the data
def generate_true_duplicates(data):
    data = data.reset_index(drop=True)
    duplicates = []

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data['product_id'][i] == data['product_id'][j]:
                duplicates.append((i, j))

    return duplicates


# Create train and test data by splitting products into training and testing sets
def create_train_and_test_data(products):
    indices = list(range(len(products)))
    train_indices = np.random.choice(indices, size=len(products), replace=True)
    test_indices = list(set(indices) - set(train_indices))

    train_products = products.iloc[train_indices]
    test_products = products.iloc[test_indices]

    return train_indices, train_products, test_indices, test_products


# Run the bootstrap evaluation with different band configurations
def bootstrap(binary_matrix, product_data, bootstraps, bands_list, number_of_hashes):
    bootstrap_results = []

    # Iterate over the different numbers of bands
    for bands in bands_list:
        print(f"Evaluating for {bands} bands...")
        rows_per_band = number_of_hashes // bands

        for bootstrap in range(bootstraps):
            print(f"------------------------------------\n BOOTSTRAP: {bootstrap + 1} \n------------------------------------")

            # Split the data into a train and test set
            train_indices, train_products, test_indices, test_products = create_train_and_test_data(product_data)

            train_matrix = binary_matrix[:, train_indices]

            true_duplicates = generate_true_duplicates(train_products)

            # Perform min hashing and LSH
            minhash_signature = min_hashing(train_matrix, number_of_hashes)
            buckets = perform_lsh(minhash_signature, bands, rows_per_band)
            lsh_pairs = find_pairs_in_bucket(buckets)

            results_scalability = evaluate_scalability(lsh_pairs, true_duplicates, product_data)

            dissimilarity = dissimilarity_matrix(lsh_pairs, train_matrix, product_data)
            cluster_pairs_train, _ = cluster_products(dissimilarity, 0.5)

            # Calculate recall, precision, and F1 score
            recall, precision = recall_and_precision(true_duplicates, cluster_pairs_train)
            f1 = f1_score(recall, precision)

            # Store the metrics for this band configuration and bootstrap iteration
            bootstrap_results.append({
                "Bands": bands,
                "Bootstrap": bootstrap + 1,
                "Fraction": results_scalability[0],
                "Pair quality": results_scalability[1],
                "Pair completeness": results_scalability[2],
                "F1": f1,
                "F1*": results_scalability[3]
            })

    # Convert results to DataFrame for better analysis
    results_df = pd.DataFrame(bootstrap_results)

    return results_df


# Get all factors of a number (used for selecting number of bands)
def get_factors(number):
    factors = []
    for i in range(1, int(number ** 0.5) + 1):
        if number % i == 0:
            factors.append(i)
            if i != number // i:
                factors.append(number // i)
    return sorted(factors)


# Example usage
number_of_hashes = 1000
bands_list = get_factors(number_of_hashes)
print("Factors (Number of Bands):", bands_list)

# Running code
data = load_data("C:/Users/Beau/PycharmProjects/computerscience/TVs-all-merged.json")
titles_data = titles(data)

titles_df = pd.DataFrame(titles_data, columns=['product_id', 'title', 'shop', 'brand'])
clean_data = data_cleaning(titles_df)

model_words = extract_all_model_words(clean_data)
binary_matrix = create_binary_vector(model_words, clean_data['title'].tolist())

number_of_hashes = 1000
bands = get_factors(number_of_hashes)
number_of_bootstraps = 5

results_dataframe = bootstrap(binary_matrix, clean_data, number_of_bootstraps, bands, number_of_hashes)

print(results_dataframe)

# Extract the data
fractions = results_dataframe["Fraction"]
pair_quality = results_dataframe["Pair quality"]
pair_completeness = results_dataframe["Pair completeness"]
f1_scores = results_dataframe["F1"]
f1_star_scores = results_dataframe["F1*"]

# Pair Quality vs Fractions
plt.figure(figsize=(6, 4))
plt.plot(fractions, pair_quality, marker='o', linestyle='None', label="Pair Quality")
plt.title("Pair Quality vs Fraction of comparisons")
plt.xlabel("Fraction")
plt.ylabel("Pair Quality")
plt.grid(True)
plt.legend()
plt.show()

# Pair Completeness vs Fractions
plt.figure(figsize=(6, 4))
plt.plot(fractions, pair_completeness, marker='o', color="orange", linestyle='None', label="Pair Completeness")
plt.title("Pair Completeness vs Fraction of comparisons")
plt.xlabel("Fraction")
plt.ylabel("Pair Completeness")
plt.grid(True)
plt.legend()
plt.show()

# F1 Measure vs Fractions
plt.figure(figsize=(6, 4))
plt.plot(fractions, f1_scores, marker='o', color="green", linestyle='None', label="F1 Measure")
plt.title("F1 vs Fraction of comparisons")
plt.xlabel("Fraction")
plt.ylabel("F1")
plt.grid(True)
plt.legend()
plt.show()

# F1 Star vs Fractions
plt.figure(figsize=(6, 4))
plt.plot(fractions, f1_star_scores, marker='o', color="red", linestyle='None', label="F1 Star")
plt.title("F1* vs Fraction of comparisons")
plt.xlabel("Fraction")
plt.ylabel("F1*")
plt.grid(True)
plt.legend()
plt.show()