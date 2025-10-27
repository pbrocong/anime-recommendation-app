import pandas as pd
import json
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. 파일 경로 설정 ---
csv_base_path = "/Users/parkhyungbin/Desktop/MyAnimeList-Database-master/data"
vscode_path = "."
anime_meta_file = os.path.join(csv_base_path, "anime.csv") 
meta_json_file = os.path.join(vscode_path, "anime-offline-database.json")

# --- 2. 텍스트 정제용 함수 ---
def clean_title(title):
    if not isinstance(title, str): return ""
    title = str(title).lower()
    title = re.sub(r'[^a-z0-9\s]', '', title)
    return title.strip()

def normalize_text_list(text_list):
    if not isinstance(text_list, list): return []
    return [s.lower().strip() for s in text_list if isinstance(s, str) and s.strip()]

# --- 3. 데이터 전처리 실행 ---
def preprocess_anime_database():
    print("--- CBF용 마스터 애니 DB 구축 시작 ---")
    try:
        # (A, B, C 단계는 이전과 동일... 생략)
        print("1. 'anime.csv' 로드 중...")
        anime_csv_df = pd.read_csv(anime_meta_file, dtype={'Episodes': str})
        anime_csv_df.rename(columns={
            'Name': 'title', 'MAL_ID': 'mal_id', 'Genres': 'genres',
            'Studios': 'studios', 'Type': 'type', 'Episodes': 'episodes',
            'Score': 'score'
        }, inplace=True)
        anime_csv_df['clean_title'] = anime_csv_df['title'].apply(clean_title)
        
        print("2. 'anime-offline-database.json' 로드 중...")
        with open(meta_json_file, 'r', encoding='utf-8') as f:
            meta_json_data = json.load(f)
        items = meta_json_data.get("data", [])
        if not items: return None
        meta_json_df = pd.json_normalize(items)
        meta_json_df = meta_json_df[['title', 'tags', 'picture']] 
        meta_json_df['clean_title'] = meta_json_df['title'].apply(clean_title)
        meta_json_df.drop_duplicates(subset=['clean_title'], inplace=True)

        print("3. 'anime.csv'와 'JSON' 병합 중...")
        anime_master_df = pd.merge(
            anime_csv_df,
            meta_json_df[['clean_title', 'tags', 'picture']],
            on='clean_title',
            how='left'
        )
        
        # (D, E, F, G 단계는 이전과 동일... 생략)
        print("4. 불필요한 컬럼 제거 중...")
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

        print("5. 결측치(NaN) 처리 중...")
        anime_master_df['score'] = pd.to_numeric(anime_master_df['score'], errors='coerce')
        anime_master_df['score'].fillna(anime_master_df['score'].mean(), inplace=True)
        anime_master_df['episodes'] = pd.to_numeric(anime_master_df['episodes'], errors='coerce')
        anime_master_df['episodes'].fillna(1, inplace=True)
        anime_master_df['genres'].fillna('Unknown', inplace=True)
        anime_master_df['studios'].fillna('Unknown', inplace=True)
        anime_master_df['tags'] = anime_master_df['tags'].apply(lambda x: x if isinstance(x, list) else [])
        placeholder_image = "https://via.placeholder.com/225x318.png?text=No+Image"
        anime_master_df['picture'].fillna(placeholder_image, inplace=True)

        print("6. 텍스트 정규화 (리스트로 변환) 중...")
        anime_master_df['genres_list'] = anime_master_df['genres'].apply(
            lambda x: [s.strip() for s in str(x).split(',')]
        ).apply(normalize_text_list)
        anime_master_df['studios_list'] = anime_master_df['studios'].apply(
            lambda x: [s.strip() for s in str(x).split(',')]
        ).apply(normalize_text_list)
        anime_master_df['tags_list'] = anime_master_df['tags'].apply(
            normalize_text_list
        )
        anime_master_df.drop(columns=['genres', 'studios', 'tags'], inplace=True)

        print("\n--- [최종 'anime_master_df' 전처리 완료] ---")
        anime_master_df.info()
        return anime_master_df

    except FileNotFoundError as e:
        print(f"\n[치명적 오류] 파일을 찾을 수 없습니다: {e.filename}")
        return None
    except Exception as e:
        print(f"\n[치명적 오류] 알 수 없는 오류 발생: {e}")
        return None

# --- [ ★★★ 구조 수정 ★★★ ] ---
# (create_feature_matrix 함수를 if __name__ == "__main__" 블록 *앞*으로 이동)
def create_feature_matrix(df):
    """전처리된 마스터 DB에서 CBF용 피처 행렬(X_cbf)을 생성합니다."""
    print("\n--- [피처 엔지니어링 시작] ---")
    
    # (A, B, C, D 단계는 이전과 동일... 생략)
    print("1. 숫자 피처 스케일링 (Score, Episodes)...")
    scaler = MinMaxScaler()
    numerical_features = scaler.fit_transform(df[['score', 'episodes']])
    numerical_features_sparse = csr_matrix(numerical_features)

    print("2. 카테고리 피처 원-핫 인코딩 (Type)...")
    encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_features_sparse = encoder.fit_transform(df['type'].values.reshape(-1, 1))

    print("3. 텍스트 피처 TF-IDF (Genres, Studios, Tags)...")
    df['genres_str'] = df['genres_list'].apply(lambda x: ' '.join(x))
    df['studios_str'] = df['studios_list'].apply(lambda x: ' '.join(x))
    df['tags_str'] = df['tags_list'].apply(lambda x: ' '.join(x))

    tfidf_genres = TfidfVectorizer(min_df=2)
    tfidf_features_genres = tfidf_genres.fit_transform(df['genres_str'])
    
    tfidf_studios = TfidfVectorizer(min_df=2)
    tfidf_features_studios = tfidf_studios.fit_transform(df['studios_str'])

    tfidf_tags = TfidfVectorizer(max_features=1000)
    tfidf_features_tags = tfidf_tags.fit_transform(df['tags_str'])

    print("\n4. 모든 피처를 하나의 희소 행렬(X_cbf)로 결합...")
    X_cbf = hstack([
        numerical_features_sparse,
        categorical_features_sparse,
        tfidf_features_genres,
        tfidf_features_studios,
        tfidf_features_tags
    ])
    
    print("\n--- [피처 엔지니어링 완료] ---")
    print(f"최종 CBF 피처 행렬 (X_cbf) 형태: {X_cbf.shape}")
    
    vectorizers = {
        'scaler': scaler, 'encoder': encoder, 'tfidf_genres': tfidf_genres,
        'tfidf_studios': tfidf_studios, 'tfidf_tags': tfidf_tags
    }
    return X_cbf, vectorizers

# --- [ ★★★ 구조 수정 ★★★ ] ---
# (중복된 if __name__ == "__main__" 블록을 하나로 통합)
if __name__ == "__main__":
    
    # --- 1. CBF 전처리 ---
    anime_master_df = preprocess_anime_database()
    
    if anime_master_df is not None:
        
        # --- 2. CBF 피처 엔지니어링 ---
        X_cbf, vectorizers = create_feature_matrix(anime_master_df)
        
        # --- 3. CBF 모델 생성 ---
        print("\n--- [CBF 모델 생성 시작] ---")
        print("1. 코사인 유사도 행렬 계산 중...")
        cosine_sim = cosine_similarity(X_cbf, X_cbf)
        print(f"  -> 코사인 유사도 행렬 생성 완료: {cosine_sim.shape}")

        # --- 4. 연관 규칙(Apriori) 생성 ---
        print("\n--- [연관 규칙(Apriori) 모델 생성 시작] ---")
        rules = None # rules 변수 초기화
        try:
            from mlxtend.preprocessing import TransactionEncoder
            from mlxtend.frequent_patterns import apriori, association_rules

            print("1. 연관 규칙 트랜잭션 인코딩 중...")
            transactions = anime_master_df['genres_list'].tolist()
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            apriori_df = pd.DataFrame(te_ary, columns=te.columns_)

            print("2. Apriori 실행 중 (min_support=0.01)...")
            frequent_itemsets = apriori(
                apriori_df, min_support=0.01, use_colnames=True
            )

            print("3. 연관 규칙 생성 중 (confidence > 0.6)...")
            rules = association_rules(
                frequent_itemsets, metric="confidence", min_threshold=0.6
            )
            print(f"  -> 연관 규칙 {len(rules)}개 생성 완료.")

        except ImportError:
            print("\n[경고] 'mlxtend' 라이브러리가 없습니다.")
        except Exception as e:
            print(f"\n[오류] 연관 규칙 생성 중 오류 발생: {e}")
        
        # --- 5. 모델 '굽기' ---
        print("\n--- [CBF/Apriori 모델 '굽기' 시작] ---")
        try:
            import joblib 
            joblib.dump(cosine_sim, 'cosine_sim_matrix.joblib')
            print("1. 'cosine_sim_matrix.joblib' 저장 완료.")

            joblib.dump(anime_master_df, 'anime_master_df.joblib')
            print("2. 'anime_master_df.joblib' 저장 완료.")
            
            if rules is not None:
                joblib.dump(rules, 'apriori_rules.joblib')
                print("3. 'apriori_rules.joblib' 저장 완료.")
            
            print("\n--- [모델 '굽기' 완료!] ---")

        except Exception as e:
            print(f"\n[오류] 모델 저장 중 오류 발생: {e}")