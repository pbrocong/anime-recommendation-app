import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict

# --- [Precision@K, Recall@K 계산 함수] ---
def precision_recall_at_k(predictions, k=10, threshold=7.0):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
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

    avg_precision = sum(p for p in precisions.values()) / len(precisions)
    avg_recall = sum(r for r in recalls.values()) / len(recalls)
    return avg_precision, avg_recall

# --- 1. [데이터 로드] ---
# (본인의 rating.csv 경로로 수정하세요!)
RATING_CSV_PATH = '/Users/parkhyungbin/Desktop/MyAnimeList-Database-master/data/rating.csv'

print(f"'{RATING_CSV_PATH}' 로드 중...")
try:
    df = pd.read_csv(RATING_CSV_PATH)
except FileNotFoundError:
    print(f"[오류] '{RATING_CSV_PATH}' 파일을 찾을 수 없습니다.")
    exit()

# -1 평점 제거 (시청했지만 평점 안 준 경우)
df = df[df['rating'] != -1]

# (데이터가 너무 크면 여기서 샘플링)
# df = df.sample(n=1000000)

reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df[['user_id', 'anime_id', 'rating']], reader)

# --- 2. [Train/Test 분리] ---
print("데이터셋을 Train (75%) / Test (25%)로 분리합니다...")
trainset, testset = train_test_split(data, test_size=0.25)

# --- 3. [모델 훈련 (SVD)] ---
print("CF 모델 (SVD)을 훈련합니다...")
model = SVD()
model.fit(trainset)

# --- 4. [예측] ---
print("Testset으로 예측을 수행합니다...")
predictions = model.test(testset)

# --- 5. [성능 지표 계산] ---
print("\n--- [CF 모델 성능 검증 결과] ---")

# 5-1. RMSE, MAE (제안서 지표)
rmse = accuracy.rmse(predictions, verbose=True)
mae = accuracy.mae(predictions, verbose=True)

# 5-2. Precision@K, Recall@K (제안서 지표)
print("\nPrecision@10, Recall@10 계산 중 (기준: 7점 이상)...")
k_value = 10
rating_threshold = 7.0
p_at_k, r_at_k = precision_recall_at_k(predictions, k=k_value, threshold=rating_threshold)

print(f"\n평점 기준: {rating_threshold}점 이상")
print(f"Precision@{k_value}: {p_at_k:.4f}")
print(f"Recall@{k_value}   : {r_at_k:.4f}")