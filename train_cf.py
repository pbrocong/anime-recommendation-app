import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans
import joblib
import os

print("--- [CF 모델 훈련 시작] ---")

# --- 1. 파일 경로 설정 ---
csv_base_path = "/Users/parkhyungbin/Desktop/MyAnimeList-Database-master/data"
RATING_CSV_PATH = os.path.join(csv_base_path, "rating.csv") # <-- 새 rating.csv
ANIME_CSV_PATH = os.path.join(csv_base_path, "anime.csv") # <-- ID/Title 매핑용

# --- 2. 데이터 로드 ---
print(f"1. '{RATING_CSV_PATH}' 로드 중... (시간이 걸릴 수 있습니다)")
try:
    df_ratings = pd.read_csv(RATING_CSV_PATH)
except FileNotFoundError:
    print(f"[오류] '{RATING_CSV_PATH}' 파일을 찾을 수 없습니다.")
    exit()

# (데이터가 너무 클 경우 샘플링)
# df_ratings = df_ratings.sample(n=1000000) 

print(f"  -> {len(df_ratings)}개 평점 로드 완료.")

# Surprise 라이브러리용 Reader (평점 범위 1-10)
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df_ratings[['user_id', 'anime_id', 'rating']], reader)

# 전체 데이터셋으로 훈련셋 구축
trainset = data.build_full_trainset()
print("2. Surprise 훈련셋 구축 완료.")

# --- 3. 모델 정의 (Item-Based k-NN) ---
print("3. Item-Based k-NN (Cosine 유사도) 모델 구성 중...")
sim_options = {
    'name': 'cosine',
    'user_based': False  # False = Item-Based
}
# KNNWithMeans: 아이템의 평균 평점을 고려하는 k-NN
model = KNNWithMeans(sim_options=sim_options, verbose=True)

# --- 4. 모델 훈련 ---
print("4. CF 모델 훈련 중... (시간이 매우 오래 걸릴 수 있습니다)")
model.fit(trainset)
print("  -> 모델 훈련 완료!")

# --- 5. ID <-> Title 매핑 생성 ---
print(f"5. '{ANIME_CSV_PATH}'에서 ID-Title 맵 생성 중...")
df_anime = pd.read_csv(ANIME_CSV_PATH, usecols=['MAL_ID', 'Name'])
# (MAL_ID -> Title) 딕셔너리 생성
id_to_title = df_anime.set_index('MAL_ID')['Name'].to_dict()
# (Title -> MAL_ID) 딕셔너리 생성
title_to_id = {v: k for k, v in id_to_title.items()}

cf_data = {
    'model': model,
    'id_to_title': id_to_title,
    'title_to_id': title_to_id,
    'trainset': trainset # raw_id <-> inner_id 변환에 필요
}
print("  -> ID-Title 맵 생성 완료.")

# --- 6. 모델 '굽기' ---
print("6. CF 모델 '굽기' 중...")
joblib.dump(cf_data, 'cf_model_data.joblib')
print("  -> 'cf_model_data.joblib' 저장 완료!")
print("\n--- [CF 모델 훈련 및 저장 완료] ---")