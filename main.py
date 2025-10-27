import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging # 로깅 추가
from fastapi.staticfiles import StaticFiles # 정적 파일(CSS, JS) 서빙
from starlette.responses import FileResponse # index.html 서빙
import os # 파일 경로 확인용
from google.cloud import storage
from google.oauth2 import service_account
# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- 0. 응답 모델 정의 (Pydantic) ---
class AnimeInfo(BaseModel):
    mal_id: Optional[int] = None # int가 아닐 수도 있으므로 Optional
    title: Optional[str] = None
    score: Optional[float] = None
    episodes: Optional[int] = None # int가 아닐 수도 있으므로 Optional
    type: Optional[str] = None
    picture: Optional[str] = None
    genres_list: List[str] = []
    studios_list: List[str] = []
    tags_list: List[str] = []
    # alias를 사용하여 Python 변수 이름과 JSON 키 이름을 다르게 설정
    Japanese_name: Optional[str] = Field(None, alias="Japanese name")
    English_name: Optional[str] = Field(None, alias="English name")
    Members: Optional[str] = None # 자동완성 정렬용

    class Config:
        # Pydantic v2 이상: allow_population_by_field_name=True
        # Pydantic v1: allow_population_by_alias=True (버전에 맞게 선택)
        allow_population_by_field_name = True # 필드 이름(Japanese_name)으로도 값 할당 허용
        # 또는 아래 줄 사용 (Pydantic v1)
        # allow_population_by_alias = True


class RecommendationItem(BaseModel):
    title: str
    similarity_score: str
    common_genres: List[str]
    common_tags_count: int

class RecommendResponse(BaseModel):
    query_title: str
    main_anime: Optional[AnimeInfo] = None
    recommendations: List[RecommendationItem] = []
    error: Optional[str] = None

# --- 1. 모델 로드 ---
models = {}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "downloaded_models")

def load_models():
    """앱 시작 시 GCS에서 모델 파일을 다운로드하고 로드합니다."""
    logger.info(" 모델 로드 중...")

    # --- [ GCS 다운로드 로직 (이전과 동일) ] ---
    BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
    CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not BUCKET_NAME or not CREDENTIALS_PATH or not os.path.exists(CREDENTIALS_PATH):
        logger.error(" [치명적 오류] GCS 환경 변수가 잘못 설정되었거나 키 파일이 없습니다.")
        exit()

    MODEL_FILES = [
        "cosine_sim_matrix.joblib", "anime_master_df.joblib",
        "apriori_rules.joblib", "cf_model_data.joblib"
    ]
    logger.info(f"모델 저장 경로: {MODEL_DIR}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    all_downloads_ok = True
    for filename in MODEL_FILES:
        local_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(local_path):
            logger.info(f"'{filename}' 다운로드 시도...")
            if not download_blob(BUCKET_NAME, filename, local_path, CREDENTIALS_PATH):
                if filename != "apriori_rules.joblib":
                     logger.error(f" [치명적 오류] 필수 모델 '{filename}' 다운로드 실패.")
                     all_downloads_ok = False # 실패 플래그 설정
                else:
                     logger.warning(" apriori_rules.joblib 다운로드 실패.")
        else:
            logger.info(f"'{filename}' 로컬 파일 사용.")
    # ---------------------------

    # --- [ ★★★ 여기가 핵심 수정 부분 ★★★ ] ---
    # 로컬 경로(MODEL_DIR)에서 모델 로드

    try:
        # 각 joblib.load() 호출 시 os.path.join(MODEL_DIR, ...) 사용
        cosine_sim_path = os.path.join(MODEL_DIR, 'cosine_sim_matrix.joblib')
        if not os.path.exists(cosine_sim_path) and not all_downloads_ok: raise FileNotFoundError(cosine_sim_path) # 다운로드 실패 시 에러 발생
        models['cosine_sim_cbf'] = joblib.load(cosine_sim_path)

        master_df_path = os.path.join(MODEL_DIR, 'anime_master_df.joblib')
        if not os.path.exists(master_df_path) and not all_downloads_ok: raise FileNotFoundError(master_df_path)
        models['anime_master_df'] = joblib.load(master_df_path)

        apriori_path = os.path.join(MODEL_DIR, 'apriori_rules.joblib')
        if os.path.exists(apriori_path):
             models['apriori_rules'] = joblib.load(apriori_path)
             logger.info(" * [Apriori 규칙 로드 완료]")
        else:
             models['apriori_rules'] = pd.DataFrame()
             logger.warning(" [경고] Apriori 규칙 파일 없음.")

        logger.info(" * [CBF 모델 로드 완료]")

        cf_data_path = os.path.join(MODEL_DIR, 'cf_model_data.joblib')
        if not os.path.exists(cf_data_path) and not all_downloads_ok: raise FileNotFoundError(cf_data_path)
        cf_data = joblib.load(cf_data_path)
        models['cf_model'] = cf_data['model']
        models['cf_id_to_title'] = cf_data['id_to_title']
        models['cf_title_to_id'] = cf_data['title_to_id']
        models['cf_trainset'] = cf_data['trainset']
        logger.info(" * [CF 모델 로드 완료]")

        models['cbf_indices'] = pd.Series(
            models['anime_master_df'].index,
            index=models['anime_master_df']['title']
        ).drop_duplicates()

        logger.info(f" * 모델 로드 완료. (총 {len(models['anime_master_df'])}개 애니 로드)")

    except FileNotFoundError as e:
        # joblib.load 실패 시 (다운로드 실패 포함)
        logger.error(f" [치명적 오류] 모델 파일 로드 실패: {e.filename}. GCS 다운로드가 성공했는지 확인하세요.")
        exit() # 로드 실패 시 서버 종료
    except Exception as e:
        logger.error(f" [치명적 오류] 모델 로드 중 예외 발생: {e}", exc_info=True)
        exit() # 로드 실패 시 서버 종료


# --- 2. FastAPI 앱 생성 및 설정 ---
app = FastAPI(title="애니메이션 추천 API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- 3. [ ★★★ 오류 수정 ★★★ ] 제목 검색 함수 ---
def find_anime_by_title(title_query: str) -> tuple[Optional[Dict[str, Any]], Optional[int]]:
    """쿼리와 일치하는 애니의 '정보'(딕셔너리)와 '인덱스'를 찾습니다."""
    df = models['anime_master_df']
    logger.info(f"제목 검색 시작: '{title_query}'")
    try:
        # 검색할 컬럼 목록 (존재하는 컬럼만 필터링)
        search_cols = ['title', 'Japanese name', 'English name']
        valid_search_cols = [col for col in search_cols if col in df.columns]
        
        if not valid_search_cols:
             logger.error("[오류] 검색할 제목 관련 컬럼('title', 'Japanese name', 'English name')이 DB에 없습니다.")
             return None, None

        masks = []
        for col in valid_search_cols:
             # na=False 추가: NaN 값이 있는 행은 검색에서 제외
             masks.append(df[col].astype(str).str.contains(title_query, case=False, na=False))

        # 모든 마스크를 OR 조건으로 결합
        combined_mask = pd.concat(masks, axis=1).any(axis=1)
        combined_matches = df[combined_mask]
        
        # mal_id 기준으로 중복 제거 (mal_id가 없으면 인덱스 기준)
        subset_col = 'mal_id' if 'mal_id' in combined_matches.columns else combined_matches.index.name
        combined_matches = combined_matches.loc[~combined_matches.index.duplicated(keep='first')]
        if subset_col == 'mal_id':
             combined_matches = combined_matches.drop_duplicates(subset=[subset_col])


    except Exception as e:
        logger.error(f"[오류] 제목 검색 중 예외 발생: {e}", exc_info=True)
        return None, None

    if combined_matches.empty:
        logger.warning(f"Query '{title_query}' -> 일치 항목 없음")
        return None, None

    # 가장 첫 번째 매칭 결과 선택
    match = combined_matches.iloc[0]
    anime_info_raw = match.to_dict() # Pandas Series를 딕셔너리로
    idx = match.name # DataFrame의 원본 인덱스

    # [수정] NaN 값을 None으로 안전하게 변환 (리스트는 그대로 둠)
    anime_info = {}
    for k, v in anime_info_raw.items():
        if isinstance(v, (list, np.ndarray)):
            # 리스트나 NumPy 배열은 그대로 유지
            anime_info[k] = v
        elif pd.isna(v):
            # 다른 타입의 NaN 값은 None으로 변환
            anime_info[k] = None
        else:
            # 그 외의 값은 그대로 유지
            anime_info[k] = v

    # 필요한 리스트 필드가 없으면 빈 리스트로 초기화 (Pydantic 모델 호환성)
    for list_key in ['genres_list', 'studios_list', 'tags_list']:
        if list_key not in anime_info or anime_info[list_key] is None:
             anime_info[list_key] = []
        # 리스트가 아닌 경우 빈 리스트로 (예: NaN 이었던 경우)
        elif not isinstance(anime_info[list_key], list):
             anime_info[list_key] = []


    logger.info(f"Query '{title_query}' -> Matched '{anime_info.get('title', 'N/A')}' (Index: {idx})")
    return anime_info, idx

# --- 4. [CBF/Hybrid] 추천 함수 ---
def get_cbf_hybrid_recommendations(anime_info: Dict[str, Any], idx: int, mode: str) -> List[Dict[str, Any]]:
    df = models['anime_master_df']
    cosine_sim = models['cosine_sim_cbf']
    rules = models.get('apriori_rules') # 규칙이 없을 수도 있음

    # anime_info가 None이거나 필요한 키가 없는 경우 빈 리스트 반환
    if not anime_info or 'genres_list' not in anime_info or 'tags_list' not in anime_info:
        logger.warning("[CBF/Hybrid] 입력 anime_info가 유효하지 않습니다.")
        return []

    source_genres = set(anime_info.get('genres_list', []))
    source_tags = set(anime_info.get('tags_list', []))

    try:
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores_top_50 = sim_scores[1:51]
        final_scores = sim_scores_top_50

        # Hybrid 모드 및 규칙 존재 여부 확인
        if mode == 'hybrid' and rules is not None and not rules.empty:
            boost_genres = set()
            for _, rule in rules.iterrows():
                antecedents = set(rule['antecedents'])
                consequents = set(rule['consequents'])
                if antecedents.issubset(source_genres):
                    boost_genres.update(consequents)

            if boost_genres:
                logger.info(f"  -> [Hybrid Boost] '{boost_genres}' 장르에 가중치 적용...")
                reranked_scores = []
                for (rec_idx, score) in sim_scores_top_50:
                    # rec_idx가 df의 인덱스 범위를 벗어나는지 확인
                    if rec_idx < len(df):
                        rec_anime_genres = set(df.loc[rec_idx].get('genres_list', []))
                        if boost_genres.intersection(rec_anime_genres):
                            score *= 1.5
                        reranked_scores.append((rec_idx, score))
                    else:
                         logger.warning(f"[Hybrid] 추천 인덱스 {rec_idx}가 범위를 벗어납니다.")
                final_scores = sorted(reranked_scores, key=lambda x: x[1], reverse=True)

        recommendations_list = []
        for (rec_idx, score) in final_scores[:10]:
             # rec_idx가 df의 인덱스 범위를 벗어나는지 확인
            if rec_idx < len(df):
                rec_anime = df.loc[rec_idx]
                common_genres = source_genres.intersection(set(rec_anime.get('genres_list', [])))
                common_tags = source_tags.intersection(set(rec_anime.get('tags_list', [])))
                recommendations_list.append({
                    "title": rec_anime.get('title', '제목 없음'),
                    "similarity_score": f"{score * 100:.2f}",
                    "common_genres": list(common_genres),
                    "common_tags_count": len(common_tags)
                })
            else:
                 logger.warning(f"[CBF/Hybrid] 최종 추천 인덱스 {rec_idx}가 범위를 벗어납니다.")
        return recommendations_list

    except IndexError:
         logger.error(f"[오류] CBF/Hybrid 계산 중 인덱스 오류 발생. 입력 인덱스: {idx}", exc_info=True)
         return []
    except Exception as e:
         logger.error(f"[오류] CBF/Hybrid 계산 중 예외 발생: {e}", exc_info=True)
         return []


# --- 5. [CF] 추천 함수 ---
def get_cf_recommendations(anime_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    cf_model = models.get('cf_model')
    cf_trainset = models.get('cf_trainset')
    cf_id_to_title = models.get('cf_id_to_title')

    # CF 모델 관련 데이터가 로드되지 않았으면 빈 리스트 반환
    if not all([cf_model, cf_trainset, cf_id_to_title]):
        logger.warning("[CF] 모델 데이터가 로드되지 않아 CF 추천을 건너<0xEB><0x9A><0x8D>니다.")
        return []

    try:
        mal_id = anime_info.get('mal_id')
        if mal_id is None:
            logger.warning("[CF] 입력 anime_info에 'mal_id'가 없습니다.")
            return []

        # inner_id 변환 시도
        try:
             inner_id = cf_trainset.to_inner_iid(mal_id)
        except ValueError:
             # Trainset에 해당 raw_id가 없는 경우
             logger.warning(f"  -> [CF] MAL ID '{mal_id}'는 CF Trainset에 없어 추천이 불가능합니다.")
             return []

        neighbors_inner_ids = cf_model.get_neighbors(inner_id, k=10)
        neighbors_mal_ids = [cf_trainset.to_raw_iid(iid) for iid in neighbors_inner_ids]

        recommendations_list = []
        for rec_mal_id in neighbors_mal_ids:
            recommendations_list.append({
                "title": cf_id_to_title.get(rec_mal_id, f"ID:{rec_mal_id} 제목 없음"),
                "similarity_score": "N/A",
                "common_genres": ["CF 추천 (유사 유저 평점 기반)"],
                "common_tags_count": 0
            })
        return recommendations_list

    except Exception as e:
        logger.error(f"  -> [CF] 추천 생성 중 오류 발생: {e}", exc_info=True)
        return []

# --- 6. API 엔드포인트 (추천) ---
@app.get("/recommend", response_model=RecommendResponse)
async def recommend_anime(
    title: str = Query(..., description="검색할 애니메이션 제목"),
    mode: str = Query('cbf', description="추천 모드 ('cbf', 'hybrid', 'cf')")
):
    logger.info(f"추천 요청 수신: title='{title}', mode='{mode}'")
    anime_info, cbf_idx = find_anime_by_title(title)

    if anime_info is None or cbf_idx is None:
        logger.warning(f"일치하는 제목 없음: '{title}'")
        return RecommendResponse(query_title=title, error='일치하는 제목을 찾을 수 없습니다.')

    recommendations = []
    try:
        if mode == 'cbf' or mode == 'hybrid':
            logger.info(f"  -> [Mode: {mode}] CBF/Hybrid 로직 실행...")
            recommendations = get_cbf_hybrid_recommendations(anime_info, cbf_idx, mode)
        elif mode == 'cf':
            logger.info("  -> [Mode: cf] CF 로직 실행...")
            recommendations = get_cf_recommendations(anime_info)
        else:
            logger.error(f"알 수 없는 모드 요청: '{mode}'")
            raise HTTPException(status_code=400, detail="알 수 없는 모드입니다. 'cbf', 'hybrid', 'cf' 중 하나를 선택하세요.")

        # Pydantic 모델로 변환
        main_anime_info = AnimeInfo(**anime_info)
        valid_recommendations = [RecommendationItem(**rec) for rec in recommendations]

        logger.info(f"추천 생성 완료: {len(valid_recommendations)}개")
        return RecommendResponse(
            query_title=title,
            main_anime=main_anime_info,
            recommendations=valid_recommendations
        )
    except Exception as e:
         logger.error(f"추천 처리 중 심각한 오류 발생: {e}", exc_info=True)
         # Pydantic 모델로 에러 응답 반환
         return RecommendResponse(query_title=title, error=f"서버 내부 오류 발생: {e}")


# --- 7. API 엔드포인트 (자동완성) ---
@app.get("/suggest", response_model=List[str])
async def suggest_anime(
    q: str = Query('', description="자동완성 검색어")
):
    if len(q) < 1: return []
    df = models['anime_master_df']
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
        if subset_col == 'mal_id':
             combined_matches = combined_matches.drop_duplicates(subset=[subset_col])


        sort_col = 'Members' if 'Members' in combined_matches.columns else 'mal_id'
        combined_matches = combined_matches.copy()
        combined_matches['sort_col_numeric'] = pd.to_numeric(combined_matches.get(sort_col, 0), errors='coerce').fillna(0) # .get 추가
        top_5 = combined_matches.sort_values(by='sort_col_numeric', ascending=False).head(5)

        suggestions = top_5['title'].tolist()
        logger.info(f"자동완성 결과: {suggestions}")
        return suggestions
    
    
    except Exception as e:
         logger.error(f"자동완성 처리 중 오류 발생: {e}", exc_info=True)
         return [] # 오류 시 빈 리스트 반환

static_files_path = os.path.join(os.path.dirname(__file__), "dist", "assets")
if os.path.exists(static_files_path):
    app.mount("/assets", StaticFiles(directory=static_files_path), name="assets")
    logger.info(f"정적 파일 마운트: /assets -> {static_files_path}")
else:
    logger.warning(f"경고: 정적 파일 경로를 찾을 수 없습니다: {static_files_path}")
    logger.warning("React 앱 빌드 후 'dist/assets' 폴더가 FastAPI 폴더 내에 있는지 확인하세요.")


    # 2. 모든 다른 경로('/')는 React의 index.html을 반환하도록 설정
    # (React Router 등을 사용할 경우 필수)
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    index_html_path = os.path.join(os.path.dirname(__file__), "dist", "index.html")
    if os.path.exists(index_html_path):
        return FileResponse(index_html_path)
    else:
        logger.error(f"오류: 'dist/index.html' 파일을 찾을 수 없습니다: {index_html_path}")
        raise HTTPException(status_code=404, detail="React 앱 파일을 찾을 수 없습니다.")

# --- 8. 앱 시작 시 모델 로드 ---
@app.on_event("startup")
async def startup_event():
    load_models()

# --- Uvicorn 실행 명령어 (터미널에서 사용) ---
# uvicorn main:app --reload --host 0.0.0.0 --port 5001

