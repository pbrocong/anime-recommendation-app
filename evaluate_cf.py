import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict

# --- Precision@K and Recall@K ---
def precision_recall_at_k(predictions, k=10, threshold=7.0):
    """
    Compute Precision@K and Recall@K for each user and return the averages.
    threshold: rating threshold to consider an item as "relevant"
    """
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}

    for uid, ratings in user_est_true.items():
        n_rel = sum((true_r >= threshold) for (_, true_r) in ratings)
        ratings.sort(key=lambda x: x[0], reverse=True)
        n_rec_k = sum((est >= threshold) for (est, _) in ratings[:k])
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in ratings[:k]
        )

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    avg_precision = sum(precisions.values()) / len(precisions)
    avg_recall = sum(recalls.values()) / len(recalls)
    return avg_precision, avg_recall


# --- 1. Load rating data ---
RATING_CSV_PATH = '/Users/parkhyungbin/Desktop/MyAnimeList-Database-master/data/rating.csv'

print(f"Loading file: {RATING_CSV_PATH}")
try:
    df = pd.read_csv(RATING_CSV_PATH)
except FileNotFoundError:
    print(f"[Error] File not found: {RATING_CSV_PATH}")
    exit()

# Remove -1 ratings (watched but no rating)
df = df[df['rating'] != -1]

# (Optional) Sample if dataset is too large
# df = df.sample(n=1000000)

reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df[['user_id', 'anime_id', 'rating']], reader)


# --- 2. Train/Test Split ---
print("Splitting dataset into Train (75%) / Test (25%)...")
trainset, testset = train_test_split(data, test_size=0.25)


# --- 3. Train SVD CF Model ---
print("Training CF model (SVD)...")
model = SVD()
model.fit(trainset)


# --- 4. Run Predictions on Test Set ---
print("Running predictions on test set...")
predictions = model.test(testset)


# --- 5. Evaluation ---
print("\n--- CF Model Evaluation Results ---")

# RMSE and MAE
rmse = accuracy.rmse(predictions, verbose=True)
mae = accuracy.mae(predictions, verbose=True)

# Precision@K and Recall@K
print(f"\nCalculating Precision@10 and Recall@10 (Relevant if rating ≥ 7.0)...")
k_value = 10
rating_threshold = 7.0
p_at_k, r_at_k = precision_recall_at_k(predictions, k=k_value, threshold=rating_threshold)

print(f"\nRelevance threshold: rating ≥ {rating_threshold}")
print(f"Precision@{k_value}: {p_at_k:.4f}")
print(f"Recall@{k_value}   : {r_at_k:.4f}")