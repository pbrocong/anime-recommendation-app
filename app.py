import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

# --- 1. 모델 로드 ---
print(" * 모델 로드 중...")
try:
    # (CBF/Hybrid 모델 로드)
    cosine_sim_cbf = joblib.load('cosine_sim_matrix.joblib')
    anime_master_df = joblib.load('anime_master_df.joblib')
    apriori_rules = joblib.load('apriori_rules.joblib') 
    
    # CBF용 '제목 -> 인덱스' 매핑
    cbf_indices = pd.Series(
        anime_master_df.index, 
        index=anime_master_df['title']
    ).drop_duplicates()
    print(" * [CBF/Hybrid 모델 로드 완료]")
    
    # (CF 모델 로드)
    cf_data = joblib.load('cf_model_data.joblib')
    cf_model = cf_data['model']
    cf_id_to_title = cf_data['id_to_title']
    cf_title_to_id = cf_data['title_to_id']
    cf_trainset = cf_data['trainset']
    print(" * [CF 모델 로드 완료]")
    
except FileNotFoundError as e:
    print(f" [오류] 모델 파일('{e.filename}')을 찾을 수 없습니다.")
    print(" 'database.py'와 'train_cf.py'를 먼저 실행하여 모델 파일을 생성하세요.")
    exit()
except KeyError as e:
    print(f" [오류] 'anime_master_df.joblib' 파일이 오래되었습니다. '{e.args[0]}' 컬럼이 없습니다.")
    print(" 'database.py'를 다시 실행하여 'English name' 등이 포함된 새 파일을 생성하세요.")
    exit()

# --- 2. Flask 앱 생성 ---
app = Flask(__name__)
CORS(app) 

# --- [ ★★★ 구조 수정 ★★★ ] ---
# (API 엔드포인트보다 *먼저* 모든 헬퍼 함수를 정의합니다)

# --- 3. [헬퍼 1] (공통) 제목 검색 함수 ---
def find_anime_by_title(title_query):
    """(CBF DB 기준) 쿼리와 일치하는 애니의 '정보'와 '인덱스'를 찾습니다."""
    try:
        # 1. 영어 제목(title) (예: Shingeki no Kyojin)
        mask_eng = anime_master_df['title'].str.contains(title_query, case=False, na=False)
        matches_eng = anime_master_df[mask_eng]
        
        # 2. 한글/일본어 제목(Japanese name) (예: 進撃の巨人)
        mask_jpn = anime_master_df['Japanese name'].str.contains(title_query, case=False, na=False)
        matches_jpn = anime_master_df[mask_jpn]

        # 3. 영어 번역 제목(English name) (예: Attack on Titan)
        mask_eng_name = anime_master_df['English name'].str.contains(title_query, case=False, na=False)
        matches_eng_name = anime_master_df[mask_eng_name]
    
    except KeyError as e:
        print(f"[오류] '{e.args[0]}' 컬럼이 'anime_master_df.joblib'에 없습니다.")
        print("-> 'database.py'를 수정하고 다시 실행했는지 확인하세요.")
        return None, None
    
    # 4. 세 검색 결과 합치기 (중복 제거)
    combined_matches = pd.concat([matches_eng, matches_jpn, matches_eng_name]).drop_duplicates(subset=['mal_id'])

    if combined_matches.empty:
        print(f"Query '{title_query}' -> 일치 항목 없음")
        return None, None
    
    # 5. 가장 첫 번째 매칭 결과 사용
    match = combined_matches.iloc[0]
    anime_info = match.to_dict()
    idx = match.name # DataFrame의 원본 인덱스
    
    print(f"Query '{title_query}' -> Matched '{anime_info['title']}' (Index: {idx})")
    return anime_info, idx

# --- 4. [헬퍼 2] (CBF/Hybrid) 추천 함수 ---
def get_cbf_hybrid_recommendations(anime_info, idx, mode='cbf'):
    source_genres = set(anime_info['genres_list'])
    sim_scores = list(enumerate(cosine_sim_cbf[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores_top_50 = sim_scores[1:51]
    final_scores = sim_scores_top_50
    
    if mode == 'hybrid':
        boost_genres = set()
        for _, rule in apriori_rules.iterrows():
            if rule['antecedents'].issubset(source_genres):
                boost_genres.update(rule['consequents'])
        
        if boost_genres:
            print(f"  -> [Hybrid Boost] '{boost_genres}' 장르에 가중치 적용...")
            reranked_scores = []
            for (rec_idx, score) in sim_scores_top_50:
                rec_anime_genres = set(anime_master_df.loc[rec_idx]['genres_list'])
                if boost_genres.intersection(rec_anime_genres):
                    score *= 1.5 # 부스트
                reranked_scores.append((rec_idx, score))
            final_scores = sorted(reranked_scores, key=lambda x: x[1], reverse=True)

    # Top 10 선정 및 '설명' 추가
    recommendations_list = []
    source_tags = set(anime_info['tags_list'])
    for (rec_idx, score) in final_scores[:10]:
        rec_anime = anime_master_df.loc[rec_idx]
        common_genres = source_genres.intersection(set(rec_anime['genres_list']))
        common_tags = source_tags.intersection(set(rec_anime['tags_list']))
        
        recommendations_list.append({
            "title": rec_anime['title'],
            "similarity_score": f"{score * 100:.2f}",
            "common_genres": list(common_genres),
            "common_tags_count": len(common_tags)
        })
    return recommendations_list

# --- 5. [헬퍼 3] (CF) 추천 함수 ---
def get_cf_recommendations(anime_info):
    try:
        mal_id = anime_info['mal_id']
        inner_id = cf_model.trainset.to_inner_iid(mal_id)
        neighbors_inner_ids = cf_model.get_neighbors(inner_id, k=10)
        neighbors_mal_ids = [cf_model.trainset.to_raw_iid(iid) for iid in neighbors_inner_ids]
        
        recommendations_list = []
        for rec_mal_id in neighbors_mal_ids:
            recommendations_list.append({
                "title": cf_id_to_title.get(rec_mal_id, "제목 없음"),
                "similarity_score": "N/A", # CF(k-NN)는 유사도 점수를 직접 제공X
                "common_genres": ["CF 추천 (유사 유저 평점 기반)"],
                "common_tags_count": 0
            })
        return recommendations_list

    except ValueError:
        print(f"  -> [CF] '{anime_info['title']}'는 평점 데이터가 부족해 CF 추천이 불가능합니다.")
        return []
    except Exception as e:
        print(f"  -> [CF] 오류: {e}")
        return []

# --- 6. API 엔드포인트 (추천) ---
@app.route('/recommend', methods=['GET'])
def recommend():
    title_query = request.args.get('title')
    mode = request.args.get('mode', 'cbf') 
    if not title_query:
        return jsonify({'error': '제목을 입력해주세요'}), 400

    # (이제 이 함수들은 이 코드 블록 *위에* 정의되어 있습니다)
    anime_info, cbf_idx = find_anime_by_title(title_query) 

    if anime_info is None:
        return jsonify({'query_title': title_query, 'error': '일치하는 제목을 찾을 수 없습니다.'})

    if mode == 'cbf' or mode == 'hybrid':
        recommendations = get_cbf_hybrid_recommendations(anime_info, cbf_idx, mode)
    elif mode == 'cf':
        recommendations = get_cf_recommendations(anime_info)
    else:
        return jsonify({'error': '알 수 없는 모드입니다.'}), 400
        
    return jsonify({
        'query_title': title_query,
        'main_anime': anime_info,
        'recommendations': recommendations
    })

# --- 7. API 엔드포인트 (자동완성) ---
@app.route('/suggest', methods=['GET'])
def suggest():
    query = request.args.get('q', '') # ?q=...
    if len(query) < 1:
        return jsonify([])

    try:
        # 1. 영어 제목(title)
        mask_eng = anime_master_df['title'].str.contains(query, case=False, na=False)
        matches_eng = anime_master_df[mask_eng]
        
        # 2. 한글/일본어 제목(Japanese name)
        mask_jpn = anime_master_df['Japanese name'].str.contains(query, case=False, na=False)
        matches_jpn = anime_master_df[mask_jpn]

        # 3. 영어 번역 제목(English name)
        mask_eng_name = anime_master_df['English name'].str.contains(query, case=False, na=False)
        matches_eng_name = anime_master_df[mask_eng_name]
    
    except KeyError as e:
        print(f"[오류] 자동완성 중 '{e.args[0]}' 컬럼이 없습니다.")
        return jsonify([]) # 빈 리스트 반환
    
    # 4. 세 검색 결과 합치기
    combined_matches = pd.concat([matches_eng, matches_jpn, matches_eng_name]).drop_duplicates(subset=['mal_id'])

    # 5. 'Members' 수(인기도)로 정렬
    sort_col = 'Members' if 'Members' in combined_matches.columns else 'mal_id'
    combined_matches = combined_matches.copy()  # 경고 방지
    combined_matches['sort_col_numeric'] = pd.to_numeric(combined_matches[sort_col], errors='coerce').fillna(0)
    top_5 = combined_matches.sort_values(by='sort_col_numeric', ascending=False).head(5)

    # 6. 'title' 리스트로 반환
    suggestions = top_5['title'].tolist()
    return jsonify(suggestions)

# --- 8. 서버 실행 ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
 