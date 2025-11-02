import pandas as pd
import json
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

# --- 1) File Paths ---
csv_base_path = "/Users/parkhyungbin/Desktop/MyAnimeList-Database-master/data"
vscode_path = "."
anime_meta_file = os.path.join(csv_base_path, "anime.csv")
meta_json_file = os.path.join(vscode_path, "anime-offline-database.json")

# --- 2) Text Utilities ---
def clean_title(title):
    """Lowercase, remove non-alphanumerics, and trim."""
    if not isinstance(title, str):
        return ""
    title = str(title).lower()
    title = re.sub(r'[^a-z0-9\s]', '', title)
    return title.strip()

def normalize_text_list(text_list):
    """Normalize a list of strings: lowercase + strip; ignore non-strings/empties."""
    if not isinstance(text_list, list):
        return []
    return [s.lower().strip() for s in text_list if isinstance(s, str) and s.strip()]

# --- 3) Build Master Anime DataFrame for CBF ---
def preprocess_anime_database():
    print("--- Building master anime DB for CBF ---")
    try:
        # A) Load anime.csv
        print("1) Loading 'anime.csv'...")
        anime_csv_df = pd.read_csv(anime_meta_file, dtype={'Episodes': str})
        anime_csv_df.rename(columns={
            'Name': 'title', 'MAL_ID': 'mal_id', 'Genres': 'genres',
            'Studios': 'studios', 'Type': 'type', 'Episodes': 'episodes',
            'Score': 'score'
        }, inplace=True)
        anime_csv_df['clean_title'] = anime_csv_df['title'].apply(clean_title)

        # B) Load anime-offline-database.json
        print("2) Loading 'anime-offline-database.json'...")
        with open(meta_json_file, 'r', encoding='utf-8') as f:
            meta_json_data = json.load(f)
        items = meta_json_data.get("data", [])
        if not items:
            return None
        meta_json_df = pd.json_normalize(items)
        meta_json_df = meta_json_df[['title', 'tags', 'picture']]
        meta_json_df['clean_title'] = meta_json_df['title'].apply(clean_title)
        meta_json_df.drop_duplicates(subset=['clean_title'], inplace=True)

        # C) Merge CSV and JSON
        print("3) Merging CSV and JSON...")
        anime_master_df = pd.merge(
            anime_csv_df,
            meta_json_df[['clean_title', 'tags', 'picture']],
            on='clean_title',
            how='left'
        )

        # D) Drop unused columns
        print("4) Dropping unnecessary columns...")
        columns_to_drop = [
            'clean_title', 'title_y',
            'Aired', 'Premiered', 'Duration', 'Rating', 'Producers', 'Licensors',
            'Ranked', 'Popularity', 'Members', 'Favorites', 'Watching',
            'Completed', 'On-Hold', 'Dropped', 'Plan to Watch',
            'Score-10', 'Score-9', 'Score-8', 'Score-7', 'Score-6',
            'Score-5', 'Score-4', 'Score-3', 'Score-2', 'Score-1'
        ]
        existing_cols_to_drop = [col for col in columns_to_drop if col in anime_master_df.columns]
        anime_master_df.drop(columns=existing_cols_to_drop, inplace=True)
        anime_master_df.rename(columns={'title_x': 'title'}, inplace=True)

        # E) Handle missing values
        print("5) Handling missing values...")
        anime_master_df['score'] = pd.to_numeric(anime_master_df['score'], errors='coerce')
        anime_master_df['score'].fillna(anime_master_df['score'].mean(), inplace=True)
        anime_master_df['episodes'] = pd.to_numeric(anime_master_df['episodes'], errors='coerce')
        anime_master_df['episodes'].fillna(1, inplace=True)
        anime_master_df['genres'].fillna('Unknown', inplace=True)
        anime_master_df['studios'].fillna('Unknown', inplace=True)
        anime_master_df['tags'] = anime_master_df['tags'].apply(lambda x: x if isinstance(x, list) else [])
        placeholder_image = "https://via.placeholder.com/225x318.png?text=No+Image"
        anime_master_df['picture'].fillna(placeholder_image, inplace=True)

        # F) Normalize strings to lists
        print("6) Normalizing text fields (to list form)...")
        anime_master_df['genres_list'] = anime_master_df['genres'].apply(
            lambda x: [s.strip() for s in str(x).split(',')]
        ).apply(normalize_text_list)
        anime_master_df['studios_list'] = anime_master_df['studios'].apply(
            lambda x: [s.strip() for s in str(x).split(',')]
        ).apply(normalize_text_list)
        anime_master_df['tags_list'] = anime_master_df['tags'].apply(normalize_text_list)
        anime_master_df.drop(columns=['genres', 'studios', 'tags'], inplace=True)

        print("\n--- [anime_master_df] Preprocessing complete ---")
        anime_master_df.info()
        return anime_master_df

    except FileNotFoundError as e:
        print(f"\n[Fatal] File not found: {e.filename}")
        return None
    except Exception as e:
        print(f"\n[Fatal] Unknown error: {e}")
        return None

# --- Feature Engineering for CBF ---
def create_feature_matrix(df):
    """Create CBF feature matrix (X_cbf) from the preprocessed master DB."""
    print("\n--- Feature Engineering ---")

    # 1) Scale numeric features
    print("1) Scaling numeric features (score, episodes)...")
    scaler = MinMaxScaler()
    numerical_features = scaler.fit_transform(df[['score', 'episodes']])
    numerical_features_sparse = csr_matrix(numerical_features)

    # 2) One-hot encode categorical features
    print("2) One-hot encoding categorical feature (type)...")
    encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_features_sparse = encoder.fit_transform(df['type'].values.reshape(-1, 1))

    # 3) TF-IDF for text features
    print("3) TF-IDF for text features (genres, studios, tags)...")
    df['genres_str'] = df['genres_list'].apply(lambda x: ' '.join(x))
    df['studios_str'] = df['studios_list'].apply(lambda x: ' '.join(x))
    df['tags_str'] = df['tags_list'].apply(lambda x: ' '.join(x))

    tfidf_genres = TfidfVectorizer(min_df=2)
    tfidf_features_genres = tfidf_genres.fit_transform(df['genres_str'])

    tfidf_studios = TfidfVectorizer(min_df=2)
    tfidf_features_studios = tfidf_studios.fit_transform(df['studios_str'])

    tfidf_tags = TfidfVectorizer(max_features=1000)
    tfidf_features_tags = tfidf_tags.fit_transform(df['tags_str'])

    # 4) Combine all into a single sparse matrix
    print("\n4) Combining all features into a single sparse matrix (X_cbf)...")
    X_cbf = hstack([
        numerical_features_sparse,
        categorical_features_sparse,
        tfidf_features_genres,
        tfidf_features_studios,
        tfidf_features_tags
    ])

    print("\n--- Feature Engineering Complete ---")
    print(f"Final CBF feature matrix shape (X_cbf): {X_cbf.shape}")

    vectorizers = {
        'scaler': scaler,
        'encoder': encoder,
        'tfidf_genres': tfidf_genres,
        'tfidf_studios': tfidf_studios,
        'tfidf_tags': tfidf_tags
    }
    return X_cbf, vectorizers

if __name__ == "__main__":

    # 1) Preprocess for CBF
    anime_master_df = preprocess_anime_database()

    if anime_master_df is not None:

        # 2) Feature engineering (CBF)
        X_cbf, vectorizers = create_feature_matrix(anime_master_df)

        # 3) Build CBF similarity
        print("\n--- Building CBF Similarity ---")
        print("1) Computing cosine similarity matrix...")
        cosine_sim = cosine_similarity(X_cbf, X_cbf)
        print(f"-> Cosine similarity matrix computed: {cosine_sim.shape}")

        # 4) Association Rules (Apriori) — optional / omitted
        print("\n--- Building Association Rules (Apriori) ---")
        rules = None
        try:
            # (Apriori code omitted for brevity)
            pass
        except ImportError:
            print("\n[Warning] 'mlxtend' library is not installed.")
        except Exception as e:
            print(f"\n[Error] Error while generating association rules: {e}")

        # 5) Train Logistic Regression (Popularity)
        print("\n--- Training Logistic Regression (Popularity) ---")
        lr_model = None
        try:
            # Create binary target: popular if score >= 7.5
            popularity_threshold = 7.5
            y_popular = (anime_master_df['score'] >= popularity_threshold).astype(int)
            print(f"1) Target created (score >= {popularity_threshold}).")
            print(f"   -> Popular(1): {y_popular.sum()} / Not popular(0): {len(y_popular) - y_popular.sum()}")

            print("2) Fitting LogisticRegression (max_iter=1000)...")
            lr_model = LogisticRegression(max_iter=1000, random_state=42)
            lr_model.fit(X_cbf, y_popular)
            print("-> Training complete.")
        except Exception as e:
            print(f"\n[Error] Logistic Regression training failed: {e}")

        # 6) Persist models/artifacts
        print("\n--- Persisting Models/Artifacts (CBF/Apriori/LR) ---")
        try:
            import joblib
            joblib.dump(cosine_sim, 'cosine_sim_matrix.joblib')
            print("1) Saved 'cosine_sim_matrix.joblib'")

            joblib.dump(anime_master_df, 'anime_master_df.joblib')
            print("2) Saved 'anime_master_df.joblib'")

            if rules is not None:
                joblib.dump(rules, 'apriori_rules.joblib')
                print("3) Saved 'apriori_rules.joblib'")

            if lr_model is not None:
                joblib.dump(lr_model, 'lr_popularity_model.joblib')
                print("4) Saved 'lr_popularity_model.joblib'")

            # Save X_cbf used for training (required for later predictions)
            joblib.dump(X_cbf, 'X_cbf_matrix.joblib')
            print("5) Saved 'X_cbf_matrix.joblib'")

            print("\n--- Persisting Complete ---")

        except Exception as e:
            print(f"\n[Error] Failed to save models: {e}")