import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans
import joblib
import os

print("--- [Starting CF Model Training] ---")

# --- 1. File Paths ---
csv_base_path = "/Users/parkhyungbin/Desktop/MyAnimeList-Database-master/data"
RATING_CSV_PATH = os.path.join(csv_base_path, "rating.csv")   # rating dataset
ANIME_CSV_PATH = os.path.join(csv_base_path, "anime.csv")     # for ID <-> Title mapping

# --- 2. Load Rating Data ---
print(f"1. Loading '{RATING_CSV_PATH}' (this may take some time)...")
try:
    df_ratings = pd.read_csv(RATING_CSV_PATH)
except FileNotFoundError:
    print(f"[Error] File not found: {RATING_CSV_PATH}")
    exit()

# Optional: Sampling large data
# df_ratings = df_ratings.sample(n=1000000)

print(f"  -> Loaded {len(df_ratings)} rating entries.")

reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df_ratings[['user_id', 'anime_id', 'rating']], reader)

# Build full training set
trainset = data.build_full_trainset()
print("2. Built full Surprise training set.")

# --- 3. Model Definition (Item-Based k-NN) ---
print("3. Building Item-Based k-NN model (Cosine similarity)...")
sim_options = {
    'name': 'cosine',
    'user_based': False  # False = Item-based collaborative filtering
}

# KNNWithMeans considers item mean ratings for improved accuracy
model = KNNWithMeans(sim_options=sim_options, verbose=True)

# --- 4. Train the Model ---
print("4. Training CF model... (This may take a long time)")
model.fit(trainset)
print("  -> Model training complete.")

# --- 5. Create ID <-> Title Mapping ---
print(f"5. Building ID-Title mapping from '{ANIME_CSV_PATH}'...")
df_anime = pd.read_csv(ANIME_CSV_PATH, usecols=['MAL_ID', 'Name'])

id_to_title = df_anime.set_index('MAL_ID')['Name'].to_dict()
title_to_id = {v: k for k, v in id_to_title.items()}

cf_data = {
    'model': model,
    'id_to_title': id_to_title,
    'title_to_id': title_to_id,
    'trainset': trainset  # Needed for raw_id ↔ inner_id conversion
}

print("  -> ID-Title mapping built successfully.")

# --- 6. Save Trained Model ---
print("6. Saving CF model data...")
joblib.dump(cf_data, 'cf_model_data.joblib')
print("  -> Saved: 'cf_model_data.joblib'")

print("\n--- [CF Model Training and Saving Complete] ---")