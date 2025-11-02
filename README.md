# 애니 추천시스템 (Team 8)

이 프로젝트는 FastAPI(백엔드)와 React(프론트엔드)를 기반으로 한 애니메이션 추천 시스템입니다. 사용자가 애니메이션을 검색하면, 콘텐츠 기반 필터링(CBF), 협업 필터링(CF), 하이브리드 모델 등 다양한 알고리즘을 통해 개인화된 추천 목록을 제공합니다.

##  주요 기능

다중 추천 모델: 사용자가 4가지 추천 방식 중 하나를 실시간으로 선택할 수 있습니다.

콘텐츠 기반 (CBF): 애니메이션의 장르, 태그, 스튜디오 등 메타데이터 유사도(코사인 유사도)를 기반으로 추천합니다.

협업 필터링 (CF): surprise 라이브러리의 Item-Based k-NN을 사용하여, 대규모 사용자 평점 패턴을 기반으로 유사하게 평가된 아이템을 추천합니다.

하이브리드 (CBF + Apriori): CBF 결과에 장르 간 연관 규칙(Apriori)을 적용하여 추천 순위를 재조정(Reranking)합니다.

인기 예측 (CBF + LR): CBF 결과에 로지스틱 회귀(Logistic Regression)로 예측한 '인기 확률'을 결합하여 순위를 재조정합니다.

검색 및 자동완성: 한글, 영문, 일본어 제목 3개 컬럼을 동시 검색하며, 입력 시 인기도 순으로 상위 5개 검색어를 제안합니다.

상세 정보 표시: 검색하거나 클릭한 애니메이션의 포스터 이미지, 점수, 장르, 스튜디오 등의 상세 정보를 표시합니다.

추천 설명: 추천된 항목에 대해 '유사도 점수', '공통 장르', '인기 확률' 등 왜 추천되었는지에 대한 간단한 근거를 함께 제공합니다.

## 기술 스택

백엔드: FastAPI (Python), Uvicorn

프론트엔드: React (JavaScript, Vite), Tailwind CSS

데이터 처리/모델링: Pandas, Scikit-learn (TF-IDF, Logistic Regression), Surprise (KNN), Mlxtend (Apriori), Joblib

데이터 소스: MyAnimeList (Kaggle), Anime Offline Database (JSON)

(배포): Render (Web Service), Google Cloud Storage (GCS) - 모델 파일 저장용 (모델이 너무 커서 배포는 안한 상태) 

## 로컬에서 실행하기

사전 준비:

모든 Python 라이브러리가 pytorch_env (또는 사용 중인) 가상 환경에 설치되어 있어야 합니다. (pip install -r requirements.txt)

my-new-anime-app 폴더에서 npm install이 완료되어 있어야 합니다.

모든 .joblib 모델 파일 (anime_master_df.joblib, cosine_sim_matrix.joblib, cf_model_data.joblib 등 5개)이 머신러닝 폴더에 준비되어 있어야 합니다. (아래 '모델 훈련' 참고)

1. 백엔드 (FastAPI) 실행

터미널 1을 열고 머신러닝 폴더로 이동합니다.

Python 가상 환경을 활성화합니다.

conda activate pytorch_env


FastAPI 서버를 실행합니다.

uvicorn main:app --reload --host 0.0.0.0 --port 5001


2. 프론트엔드 (React) 실행

터미널 2 (새 창)를 열고 my-new-anime-app 폴더로 이동합니다.

cd my-new-anime-app


React 개발 서버를 실행합니다.

npm run dev


웹 브라우저에서 터미널에 표시된 Local 주소 (예: http://localhost:5173/)로 접속합니다.

모델 훈련 (최초 1회 필수)

main.py 서버를 실행하기 전에, 추천 모델 파일(*.joblib)들이 머신러닝 폴더에 생성되어 있어야 합니다.

실행 순서:
(pytorch_env가 활성화된 머신러닝 폴더 터미널에서 실행)

CBF/Apriori/인기예측 데이터 생성:

python database.py


(생성 파일: anime_master_df.joblib, cosine_sim_matrix.joblib, apriori_rules.joblib, feature_data.joblib)

CF 모델 훈련:

python train_cf.py


(생성 파일: cf_model_data.joblib)

주의: 대용량 rating.csv 파일을 사용하므로 시간이 매우 오래 걸릴 수 있습니다.

인기 예측 모델 훈련:

python train_popularity_model.py


(생성 파일: popularity_model.joblib)

성능 검증

CF 모델(SVD)의 RMSE, Precision@K 등 객관적인 성능 지표는 별도 스크립트로 확인할 수 있습니다.

python evaluate_cf.py
