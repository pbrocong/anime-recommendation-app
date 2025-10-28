import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
import os
# GCS 관련 import 제거

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 0. 응답 모델 정의 (Pydantic) ---
# (이전 코드와 동일... 생략)
class AnimeInfo(BaseModel):
    mal_id: Optional[int] = None; title: Optional[str] = None; score: Optional[float] = None
    episodes: Optional[int] = None; type: Optional[str] = None; picture: Optional[str] = None
    genres_list: List[str] = []; studios_list: List[str] = []; tags_list: List[str] = []
    Japanese_name: Optional[str] = Field(None, alias="Japanese name")
    English_name: Optional[str] = Field(None, alias="English name")
    Members: Optional[str] = None
    class Config: allow_population_by_field_name = True
class RecommendationItem(BaseModel): # ... (동일)
    title: str; similarity_score: str; common_genres: List[str]; common_tags_count: int
class RecommendResponse(BaseModel): # ... (동일)
    query_title: str; main_anime: Optional[AnimeInfo] = None; recommendations: List[RecommendationItem] = []; error: Optional[str] = None

# --- 1. 모델 로드 (수정됨: 로컬 파일 직접 로드) ---
models = {}
# 모델 파일이 있는 경로 (main.py와 같은 폴더)
MODEL_DIR = os.path.dirname(__file__) # 현재 파일이 있는 디렉토리

def load_models():
    """앱 시작 시 로컬 폴더에서 모델 파일을 로드합니다."""
    logger.info(" 로컬 모델 파일 로드 중...")
    try:
        # 각 모델 파일 경로 설정
        cosine_sim_path = os.path.join(MODEL_DIR, 'cosine_sim_matrix.joblib')
        master_df_path = os.path.join(MODEL_DIR, 'anime_master_df.joblib')
        apriori_path = os.path.join(MODEL_DIR, 'apriori_rules.joblib')
        cf_data_path = os.path.join(MODEL_DIR, 'cf_model_data.joblib')

        # 파일 존재 여부 확인 후 로드
        if not os.path.exists(cosine_sim_path): raise FileNotFoundError(cosine_sim_path)
        models['cosine_sim_cbf'] = joblib.load(cosine_sim_path)

        if not os.path.exists(master_df_path): raise FileNotFoundError(master_df_path)
        models['anime_master_df'] = joblib.load(master_df_path)

        if os.path.exists(apriori_path):
             models['apriori_rules'] = joblib.load(apriori_path)
             logger.info(" * [Apriori 규칙 로드 완료]")
        else:
             models['apriori_rules'] = pd.DataFrame()
             logger.warning(" [경고] Apriori 규칙 파일 없음/로드 실패.")

        logger.info(" * [CBF 모델 로드 완료]")

        if not os.path.exists(cf_data_path): raise FileNotFoundError(cf_data_path)
        cf_data = joblib.load(cf_data_path)
        models['cf_model'] = cf_data['model']
        models['cf_id_to_title'] = cf_data['id_to_title']
        models['cf_title_to_id'] = cf_data['title_to_id']
        models['cf_trainset'] = cf_data['trainset']
        logger.info(" * [CF 모델 로드 완료]")

        # '제목 -> 인덱스' 매핑 생성
        models['cbf_indices'] = pd.Series(
            models['anime_master_df'].index,
            index=models['anime_master_df']['title']
        ).drop_duplicates()

        logger.info(f" * 모델 로드 완료. (총 {len(models['anime_master_df'])}개 애니 로드)")

    except FileNotFoundError as e:
        logger.error(f" [치명적 오류] 모델 파일 로드 실패: {e.filename}.")
        logger.error(" 'database.py'와 'train_cf.py'를 실행하여 모델 파일(.joblib)을 생성했는지 확인하세요.")
        # 로컬 실행 시 에러 발생 시 종료하는 것이 더 명확할 수 있음
        raise RuntimeError(f"Model file not found: {e.filename}")
    except Exception as e:
        logger.error(f" [치명적 오류] 모델 로드 중 예외 발생: {e}", exc_info=True)
        raise RuntimeError(f"Model Loading Exception: {e}")


# --- 2. FastAPI 앱 생성 및 설정 ---
app = FastAPI(title="애니메이션 추천 API (로컬)", version="1.0") # 제목 변경 (선택 사항)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- (헬퍼 함수 find_anime_by_title, get_cbf..., get_cf... 동일 ... 생략) ---
# --- 3. 제목 검색 함수 ---
def find_anime_by_title(title_query: str) -> tuple[Optional[Dict[str, Any]], Optional[int]]:
    # ... (이전과 동일, 에러 로깅 강화 버전 유지) ...
    df = models.get('anime_master_df')
    if df is None: logger.error("[오류] anime_master_df가 로드되지 않았습니다."); return None, None
    logger.info(f"제목 검색 시작: '{title_query}'")
    try:
        search_cols = ['title', 'Japanese name', 'English name']
        valid_search_cols = [col for col in search_cols if col in df.columns]
        if not valid_search_cols: return None, None
        masks = [df[col].astype(str).str.contains(title_query, case=False, na=False) for col in valid_search_cols]
        combined_mask = pd.concat(masks, axis=1).any(axis=1)
        combined_matches = df[combined_mask]
        subset_col = 'mal_id' if 'mal_id' in combined_matches.columns else combined_matches.index.name
        combined_matches = combined_matches.loc[~combined_matches.index.duplicated(keep='first')]
        if subset_col == 'mal_id': combined_matches = combined_matches.drop_duplicates(subset=[subset_col])
    except Exception as e: logger.error(f"[오류] 제목 검색 중 예외 발생: {e}", exc_info=True); return None, None
    if combined_matches.empty: logger.warning(f"Query '{title_query}' -> 일치 항목 없음"); return None, None
    match = combined_matches.iloc[0]; anime_info_raw = match.to_dict(); idx = match.name
    anime_info = {}
    for k, v in anime_info_raw.items():
        if isinstance(v, (list, np.ndarray)): anime_info[k] = v
        elif pd.isna(v): anime_info[k] = None
        else: anime_info[k] = v
    for list_key in ['genres_list', 'studios_list', 'tags_list']:
        if list_key not in anime_info or anime_info[list_key] is None: anime_info[list_key] = []
        elif not isinstance(anime_info[list_key], list): anime_info[list_key] = []
    logger.info(f"Query '{title_query}' -> Matched '{anime_info.get('title', 'N/A')}' (Index: {idx})")
    return anime_info, idx

# --- 4. CBF/Hybrid 추천 함수 ---
def get_cbf_hybrid_recommendations(anime_info: Dict[str, Any], idx: int, mode: str) -> List[Dict[str, Any]]:
    # ... (이전과 동일, 에러 로깅 강화 버전 유지) ...
    df = models.get('anime_master_df'); cosine_sim = models.get('cosine_sim_cbf'); rules = models.get('apriori_rules')
    if df is None or cosine_sim is None: logger.error("[CBF/Hybrid] 모델 데이터 누락."); return []
    if not anime_info or 'genres_list' not in anime_info or 'tags_list' not in anime_info: return []
    source_genres = set(anime_info.get('genres_list', [])); source_tags = set(anime_info.get('tags_list', []))
    try:
        sim_scores = list(enumerate(cosine_sim[idx])); sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores_top_50 = sim_scores[1:51]; final_scores = sim_scores_top_50
        if mode == 'hybrid' and rules is not None and not rules.empty:
            boost_genres = set()
            for _, rule in rules.iterrows():
                antecedents = set(rule['antecedents']); consequents = set(rule['consequents'])
                if antecedents.issubset(source_genres): boost_genres.update(consequents)
            if boost_genres:
                logger.info(f"-> [Hybrid Boost] '{boost_genres}' 장르 가중치 적용...")
                reranked_scores = []
                for (rec_idx, score) in sim_scores_top_50:
                    if rec_idx < len(df):
                        rec_anime_genres = set(df.loc[rec_idx].get('genres_list', []))
                        if boost_genres.intersection(rec_anime_genres): score *= 1.5
                        reranked_scores.append((rec_idx, score))
                    else: logger.warning(f"[Hybrid] 인덱스 {rec_idx} 범위 초과.")
                final_scores = sorted(reranked_scores, key=lambda x: x[1], reverse=True)
        recommendations_list = []
        for (rec_idx, score) in final_scores[:10]:
            if rec_idx < len(df):
                rec_anime = df.loc[rec_idx]
                common_genres = source_genres.intersection(set(rec_anime.get('genres_list', [])))
                common_tags = source_tags.intersection(set(rec_anime.get('tags_list', [])))
                recommendations_list.append({
                    "title": rec_anime.get('title', '제목 없음'), "similarity_score": f"{score * 100:.2f}",
                    "common_genres": list(common_genres), "common_tags_count": len(common_tags)
                })
            else: logger.warning(f"[CBF/Hybrid] 최종 인덱스 {rec_idx} 범위 초과.")
        return recommendations_list
    except IndexError: logger.error(f"[오류] CBF/Hybrid 인덱스 오류. 입력 인덱스: {idx}", exc_info=True); return []
    except Exception as e: logger.error(f"[오류] CBF/Hybrid 예외 발생: {e}", exc_info=True); return []

# --- 5. CF 추천 함수 ---
def get_cf_recommendations(anime_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    # ... (이전과 동일, 에러 로깅 강화 버전 유지) ...
    cf_model = models.get('cf_model'); cf_trainset = models.get('cf_trainset'); cf_id_to_title = models.get('cf_id_to_title')
    if not all([cf_model, cf_trainset, cf_id_to_title]): logger.warning("[CF] 모델 데이터 누락."); return []
    try:
        mal_id = anime_info.get('mal_id');
        if mal_id is None: logger.warning("[CF] 'mal_id' 없음."); return []
        try: inner_id = cf_trainset.to_inner_iid(mal_id)
        except ValueError: logger.warning(f"-> [CF] MAL ID '{mal_id}'는 Trainset에 없음."); return []
        neighbors_inner_ids = cf_model.get_neighbors(inner_id, k=10)
        neighbors_mal_ids = [cf_trainset.to_raw_iid(iid) for iid in neighbors_inner_ids]
        recommendations_list = []
        for rec_mal_id in neighbors_mal_ids:
            recommendations_list.append({
                "title": cf_id_to_title.get(rec_mal_id, f"ID:{rec_mal_id} 제목 없음"), "similarity_score": "N/A",
                "common_genres": ["CF 추천 (유사 유저 평점 기반)"], "common_tags_count": 0
            })
        return recommendations_list
    except Exception as e: logger.error(f"-> [CF] 추천 생성 중 오류: {e}", exc_info=True); return []


# --- 6. API 엔드포인트 (추천) ---
@app.get("/recommend", response_model=RecommendResponse)
async def recommend_anime(
    title: str = Query(..., description="검색할 애니메이션 제목"),
    mode: str = Query('cbf', description="추천 모드 ('cbf', 'hybrid', 'cf')")
):
    # ... (이전과 동일, 오류 로깅 강화 버전 유지) ...
    logger.info(f"추천 요청 수신: title='{title}', mode='{mode}'")
    anime_info, cbf_idx = find_anime_by_title(title)
    if anime_info is None or cbf_idx is None:
        logger.warning(f"일치하는 제목 없음: '{title}'")
        return RecommendResponse(query_title=title, error='일치하는 제목을 찾을 수 없습니다.')
    recommendations = []
    try:
        if mode == 'cbf' or mode == 'hybrid':
            logger.info(f"-> [Mode: {mode}] CBF/Hybrid 로직 실행...")
            recommendations = get_cbf_hybrid_recommendations(anime_info, cbf_idx, mode)
        elif mode == 'cf':
            logger.info("-> [Mode: cf] CF 로직 실행...")
            recommendations = get_cf_recommendations(anime_info)
        else:
            logger.error(f"알 수 없는 모드 요청: '{mode}'")
            raise HTTPException(status_code=400, detail="알 수 없는 모드입니다.")
        main_anime_info = AnimeInfo(**anime_info)
        valid_recommendations = [RecommendationItem(**rec) for rec in recommendations]
        logger.info(f"추천 생성 완료: {len(valid_recommendations)}개")
        return RecommendResponse(query_title=title, main_anime=main_anime_info, recommendations=valid_recommendations)
    except Exception as e:
         logger.error(f"추천 처리 중 심각한 오류 발생: {e}", exc_info=True)
         return RecommendResponse(query_title=title, error=f"서버 내부 오류 발생.")


# --- 7. API 엔드포인트 (자동완성) ---
@app.get("/suggest", response_model=List[str])
async def suggest_anime(
    q: str = Query('', description="자동완성 검색어")
):
    # ... (이전과 동일, 오류 로깅 강화 버전 유지) ...
    if len(q) < 1: return []
    df = models.get('anime_master_df');
    if df is None: return []
    logger.info(f"자동완성 요청: q='{q}'")
    try:
        search_cols = ['title', 'Japanese name', 'English name']
        valid_search_cols = [col for col in search_cols if col in df.columns]
        if not valid_search_cols: return []
        masks = [df[col].astype(str).str.contains(q, case=False, na=False) for col in valid_search_cols]
        combined_mask = pd.concat(masks, axis=1).any(axis=1)
        combined_matches = df[combined_mask]
        subset_col = 'mal_id' if 'mal_id' in combined_matches.columns else combined_matches.index.name
        combined_matches = combined_matches.loc[~combined_matches.index.duplicated(keep='first')]
        if subset_col == 'mal_id': combined_matches = combined_matches.drop_duplicates(subset=[subset_col])
        sort_col = 'Members' if 'Members' in combined_matches.columns else 'mal_id'
        combined_matches = combined_matches.copy()
        combined_matches['sort_col_numeric'] = pd.to_numeric(combined_matches.get(sort_col, 0), errors='coerce').fillna(0)
        top_5 = combined_matches.sort_values(by='sort_col_numeric', ascending=False).head(5)
        suggestions = top_5['title'].tolist()
        logger.info(f"자동완성 결과: {suggestions}")
        return suggestions
    except Exception as e:
         logger.error(f"자동완성 처리 중 오류 발생: {e}", exc_info=True)
         return []


# --- 8. 정적 파일 및 앱 시작 ---
# [ ★★★ 수정된 부분: 로컬 실행 시에는 정적 파일 서빙 불필요 ★★★ ]
# (React 개발 서버(npm run dev)가 자체적으로 index.html을 서빙하므로 아래 코드 주석 처리)
"""
static_files_path = os.path.join(os.path.dirname(__file__), "dist", "assets")
if os.path.exists(static_files_path):
     app.mount("/assets", StaticFiles(directory=static_files_path), name="assets")
     logger.info(f"정적 파일 마운트: /assets -> {static_files_path}")
else: logger.warning(f"경고: 정적 파일 경로 없음: {static_files_path}")

@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    index_html_path = os.path.join(os.path.dirname(__file__), "dist", "index.html")
    if os.path.exists(index_html_path): return FileResponse(index_html_path)
    else: logger.error(f"오류: 'dist/index.html' 파일 없음: {index_html_path}"); raise HTTPException(status_code=404)
"""

@app.on_event("startup")
async def startup_event():
    load_models() # 시작 시 모델 로드

# --- Uvicorn 실행 명령어 (터미널에서 사용) ---
# uvicorn main:app --reload --host 0.0.0.0 --port 5001

