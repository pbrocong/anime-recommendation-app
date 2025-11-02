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
# Removed GCS-related imports

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 0. Response models (Pydantic) ---
class AnimeInfo(BaseModel):
    mal_id: Optional[int] = None; title: Optional[str] = None; score: Optional[float] = None
    episodes: Optional[int] = None; type: Optional[str] = None; picture: Optional[str] = None
    genres_list: List[str] = []; studios_list: List[str] = []; tags_list: List[str] = []
    Japanese_name: Optional[str] = Field(None, alias="Japanese name")
    English_name: Optional[str] = Field(None, alias="English name")
    Members: Optional[str] = None
    class Config: allow_population_by_field_name = True
class RecommendationItem(BaseModel): # ... (same)
    title: str; similarity_score: str; common_genres: List[str]; common_tags_count: int
class RecommendResponse(BaseModel): # ... (same)
    query_title: str; main_anime: Optional[AnimeInfo] = None; recommendations: List[RecommendationItem] = []; error: Optional[str] = None

models = {}
MODEL_DIR = os.path.dirname(__file__) 

def load_models():
    """앱 시작 시 로컬 폴더에서 모델 파일을 로드합니다."""
    logger.info(" 로컬 모델 파일 로드 중...")
    try:
        # Set paths for each model file
        cosine_sim_path = os.path.join(MODEL_DIR, 'cosine_sim_matrix.joblib')
        master_df_path = os.path.join(MODEL_DIR, 'anime_master_df.joblib')
        apriori_path = os.path.join(MODEL_DIR, 'apriori_rules.joblib')
        cf_data_path = os.path.join(MODEL_DIR, 'cf_model_data.joblib')
        lr_model_path = os.path.join(MODEL_DIR, 'lr_popularity_model.joblib') # [ ★★★ added ★★★ ]
        X_cbf_path = os.path.join(MODEL_DIR, 'X_cbf_matrix.joblib') # [ ★★★ added ★★★ ]

       
        if not os.path.exists(cosine_sim_path): raise FileNotFoundError(cosine_sim_path)
        models['cosine_sim_cbf'] = joblib.load(cosine_sim_path)

        if not os.path.exists(master_df_path): raise FileNotFoundError(master_df_path)
        models['anime_master_df'] = joblib.load(master_df_path)

        if os.path.exists(apriori_path):
             models['apriori_rules'] = joblib.load(apriori_path)
             logger.info(" * [Apriori 규칙 로드 완료]")
        else:
             
             logger.warning(" [경고] Apriori 규칙 파일 없음/로드 실패.")
        logger.info(" * [CBF 모델 로드 완료]")

        
        if not os.path.exists(cf_data_path): raise FileNotFoundError(cf_data_path)
        cf_data = joblib.load(cf_data_path)
        models['cf_model'] = cf_data['model']
        
        logger.info(" * [CF 모델 로드 완료]")

        
        if os.path.exists(lr_model_path) and os.path.exists(X_cbf_path):
            models['lr_model'] = joblib.load(lr_model_path)
            models['X_cbf'] = joblib.load(X_cbf_path)
            logger.info(" * [LR 인기 예측 모델 로드 완료]")
        else:
            models['lr_model'] = None
            models['X_cbf'] = None
            logger.warning(" [경고] LR 인기 예측 모델 파일(lr_model 또는 X_cbf) 없음/로드 실패.")
       

        # Build "title -> index" mapping
        models['cbf_indices'] = pd.Series(
            models['anime_master_df'].index,
            index=models['anime_master_df']['title']
        ).drop_duplicates()

        logger.info(f" * 모델 로드 완료. (총 {len(models['anime_master_df'])}개 애니 로드)")

    except FileNotFoundError as e:
        # ( ... )
        raise RuntimeError(f"Model file not found: {e.filename}")
    except Exception as e:
        # ( ... )
        raise RuntimeError(f"Model Loading Exception: {e}")

# --- 2. FastAPI app setup ---
app = FastAPI(title="애니메이션 추천 API (로컬)", version="1.0") # Title change (optional)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- (Helper functions find_anime_by_title, get_cbf..., get_cf... same ... omitted) ---
# --- 3. Title search function ---
def find_anime_by_title(title_query: str) -> tuple[Optional[Dict[str, Any]], Optional[int]]:
    # ... (Same as before, with enhanced error logging) ...
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

# --- 4. CBF/Hybrid recommendation function ---
def get_cbf_hybrid_recommendations(anime_info: Dict[str, Any], idx: int, mode: str) -> List[Dict[str, Any]]:
   
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

# --- 5. CF recommendation function ---
def get_cf_recommendations(anime_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    
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

def get_lr_hybrid_recommendations(anime_info: Dict[str, Any], idx: int) -> List[Dict[str, Any]]:
    """CBF Top50 + LR 인기 예측 확률로 리랭킹하여 Top10 반환"""
    df = models.get('anime_master_df')
    cosine_sim = models.get('cosine_sim_cbf')
    lr_model = models.get('lr_model')
    X_cbf = models.get('X_cbf') # Feature matrix used for prediction

    if not all([df is not None, cosine_sim is not None, lr_model is not None, X_cbf is not None]):
        logger.error("[LR-Hybrid] 모델 데이터 누락 (df, cosine_sim, lr_model, X_cbf).")
        return []

    # Base info (for shared-genre computation)
    source_genres = set(anime_info.get('genres_list', []))
    source_tags = set(anime_info.get('tags_list', []))

    try:
        # 1) CBF Top-50 candidates
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores_top_50 = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:51]

        # 2) Predict popularity probability with LR
        reranked_scores = []
        
        # Extract indices and CBF scores
        top_50_indices = [i for i, score in sim_scores_top_50]
        if not top_50_indices:
            return []
            
        # Slice feature rows for Top-50 from X_cbf (CSR supports list indexing)
        X_top_50 = X_cbf[top_50_indices]
        
        # Predict class-1 probability (popularity)
        popularity_probs = lr_model.predict_proba(X_top_50)[:, 1]

        # 3) Combine CBF score and LR prob (reranking)
        for i, (rec_idx, cbf_score) in enumerate(sim_scores_top_50):
            lr_prob = popularity_probs[i]
            # Weighted combination, tunable
            # Example: 70% CBF + 30% LR prob
            combined_score = (cbf_score * 0.7) + (lr_prob * 0.3)
            reranked_scores.append((rec_idx, combined_score, cbf_score, lr_prob))

        # 4) Sort by combined score
        final_scores = sorted(reranked_scores, key=lambda x: x[1], reverse=True)

        # 5) Format Top-10
        recommendations_list = []
        for (rec_idx, combined_score, cbf_score, lr_prob) in final_scores[:10]:
            if rec_idx < len(df):
                rec_anime = df.loc[rec_idx]
                common_genres = source_genres.intersection(set(rec_anime.get('genres_list', [])))
                common_tags = source_tags.intersection(set(rec_anime.get('tags_list', [])))
                score_str = f"CBF {cbf_score * 100:.1f} (인기 {lr_prob*100:.0f}%)"
                recommendations_list.append({
                    "title": rec_anime.get('title', '제목 없음'),
                    "similarity_score": score_str, # [ ★★★ modified ★★★ ]
                    "common_genres": list(common_genres),
                    "common_tags_count": len(common_tags)
                })
            else:
                logger.warning(f"[LR-Hybrid] 최종 인덱스 {rec_idx} 범위 초과.")
        return recommendations_list

    except IndexError:
        logger.error(f"[오류] LR-Hybrid 인덱스 오류. 입력 인덱스: {idx}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"[오류] LR-Hybrid 예외 발생: {e}", exc_info=True)
        return []
    
# --- 6. API endpoint (recommendations) ---
@app.get("/recommend", response_model=RecommendResponse)
async def recommend_anime(
    title: str = Query(..., description="검색할 애니메이션 제목"),
    mode: str = Query('cbf', description="추천 모드 ('cbf', 'hybrid', 'lr_hybrid', 'cf')") # [ ★★★ modified ★★★ ]
):
    logger.info(f"추천 요청 수신: title='{title}', mode='{mode}'")
    anime_info, cbf_idx = find_anime_by_title(title)
    if anime_info is None or cbf_idx is None:
        logger.warning(f"일치하는 제목 없음: '{title}'")
        return RecommendResponse(query_title=title, error='일치하는 제목을 찾을 수 없습니다.')
    
    recommendations = []
    try:
        if mode == 'cbf' or mode == 'hybrid':
            logger.info(f"-> [Mode: {mode}] CBF/Apriori Hybrid 로직 실행...")
            recommendations = get_cbf_hybrid_recommendations(anime_info, cbf_idx, mode)
        
        # [ lr_hybrid branch ]
        elif mode == 'lr_hybrid':
            logger.info("-> [Mode: lr_hybrid] LR Hybrid 로직 실행...")
            recommendations = get_lr_hybrid_recommendations(anime_info, cbf_idx)
        # [ addition complete ]

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

# --- 7. API endpoint (autocomplete) ---
@app.get("/suggest", response_model=List[str])
async def suggest_anime(
    q: str = Query('', description="자동완성 검색어")
):
    
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

# --- 8. Static files & app start ---
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
    load_models() # Load models at startup

# --- Uvicorn run command (use in terminal) ---
# uvicorn main:app --reload --host 0.0.0.0 --port 5001