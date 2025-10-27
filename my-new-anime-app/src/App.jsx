import React, { useState, useEffect, useCallback } from 'react';

// --- API 기본 URL ---
const API_BASE_URL = 'http://127.0.0.1:5001'; // FastAPI 서버 주소

// --- SVG 아이콘 컴포넌트 ---
// 검색 로고 아이콘 (간단한 'ani' 스타일)
const SearchLogoIcon = () => (
    <svg width="28" height="28" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M20 80 V 20 H 40 V 50 H 60 V 20 H 80 V 80 H 60 V 50 H 40 V 80 H 20 Z" fill="#00A3D9"/>
    </svg>
);
// 돋보기 아이콘
const SearchIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00A3D9" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8"></circle>
        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
    </svg>
);


// --- 컴포넌트 정의 ---

// 1. 검색창 + 자동완성
function SearchBar({ onSearch }) {
    const [query, setQuery] = useState('');
    const [suggestions, setSuggestions] = useState([]);
    const [debounceTimeout, setDebounceTimeout] = useState(null);

    const fetchSuggestions = useCallback(async (currentQuery) => {
        if (currentQuery.length < 1) { setSuggestions([]); return; }
        try {
            const response = await fetch(`${API_BASE_URL}/suggest?q=${currentQuery}`);
            if (response.ok) { setSuggestions(await response.json()); }
            else { setSuggestions([]); }
        } catch (error) { console.error("Suggestion fetch error:", error); setSuggestions([]); }
    }, []);

    const handleInputChange = (event) => {
        const newQuery = event.target.value;
        setQuery(newQuery);
        if (debounceTimeout) clearTimeout(debounceTimeout);
        const newTimeout = setTimeout(() => fetchSuggestions(newQuery), 200);
        setDebounceTimeout(newTimeout);
    };

    const handleSuggestionClick = (title) => {
        setQuery(title);
        setSuggestions([]);
        onSearch(title);
    };

    const handleSearchClick = () => {
        setSuggestions([]);
        onSearch(query);
    };

    const handleKeyPress = (event) => { if (event.key === 'Enter') handleSearchClick(); };

    return (
        <div className="relative w-full max-w-xl mx-auto my-5"> {/* 상하 마진 추가 */}
            {/* 검색창 UI */}
            <div className="flex items-center h-14 border-2 border-[#00A3D9] rounded-full bg-white pl-4 pr-2 shadow-sm"> {/* 높이, 그림자 조정 */}
                <div className="mr-3"><SearchLogoIcon /></div> {/* 로고 아이콘 */}
                <input
                    type="text" value={query} onChange={handleInputChange} onKeyPress={handleKeyPress}
                    placeholder="애니 제목 (한글/영문) 입력..."
                    className="flex-grow border-none outline-none text-lg bg-transparent text-[#333]"
                    autoComplete="off"
                />
                <button
                    onClick={handleSearchClick}
                    className="h-10 w-10 flex items-center justify-center border-none bg-transparent text-[#00A3D9] cursor-pointer rounded-full hover:bg-[#E0F7FA]" /* 버튼 스타일 */
                >
                    <SearchIcon /> {/* 돋보기 아이콘 */}
                </button>
            </div>
            {/* 자동완성 드롭다운 */}
            {suggestions.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-[#B2EBF2] rounded-lg shadow-lg z-10 max-h-60 overflow-y-auto">
                    {suggestions.map((title, index) => (
                        <div
                            key={index}
                            className="px-5 py-3 text-base cursor-pointer hover:bg-[#E0F7FA]" /* 배경색 변경 */
                            onClick={() => handleSuggestionClick(title)}
                        >
                            {title}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

// 2. 추천 모드 선택
function ModeSelector({ selectedMode, onChange }) {
    const modes = [
        { value: 'cbf', label: '콘텐츠 기반' },
        { value: 'hybrid', label: '하이브리드' },
        { value: 'cf', label: '협업 필터링' },
    ];

    return (
        <div className="my-5 text-base text-[#333] text-center"> {/* 가운데 정렬 */}
            <strong className="mr-4">추천 방식:</strong>
            {modes.map(mode => (
                <label key={mode.value} className="mr-4 cursor-pointer">
                    <input
                        type="radio" name="rec_mode" value={mode.value}
                        checked={selectedMode === mode.value}
                        onChange={(e) => onChange(e.target.value)}
                        className="mr-1 accent-[#00A3D9]"
                    />
                    {mode.label}
                </label>
            ))}
        </div>
    );
}

// 3. 메인 애니 정보 카드
function AnimeInfoCard({ anime }) {
    if (!anime) {
        return (
            <div className="bg-white rounded-lg p-6 shadow-lg min-h-[400px]"> {/* 패딩, 그림자, 최소 높이 */}
                <h3 className="text-[#00A3D9] text-xl font-semibold">애니메이션 정보</h3>
                <p className="text-[#555] mt-2">궁금한 애니메이션을 검색하면 여기에 정보가 표시됩니다.</p>
            </div>
        );
    }
    const japaneseName = anime['Japanese name'] || anime['Japanese_name'];
    const englishName = anime['English name'] || anime['English_name'];

    return (
        <div className="bg-white rounded-lg p-6 shadow-lg overflow-hidden">
            <img
                src={anime.picture || "https://via.placeholder.com/225x318.png?text=No+Image"}
                alt={`${anime.title} Poster`}
                className="w-[225px] h-[318px] rounded-lg float-right ml-6 mb-4 border border-[#B2EBF2] object-cover shadow-md" /* 그림자 추가 */
                onError={(e) => { e.target.onerror = null; e.target.src="https://via.placeholder.com/225x318.png?text=Image+Error" }}
            />
            <h2 className="text-2xl font-bold text-[#00A3D9] mt-0 mb-1">{anime.title}</h2>
            <p className="text-xs text-[#555] mb-3">(MAL ID: {anime.mal_id})</p> {/* ID 폰트 크기 조정 */}
            <p className="text-sm text-[#555] mb-3 clear-right"> {/* float 해제 */}
                {japaneseName && `원제: ${japaneseName}`} {englishName && ` | 영문: ${englishName}`}
            </p>
            <p className="text-base text-[#333] mb-4">
                <strong>점수:</strong> {anime.score?.toFixed(2)} | <strong>에피소드:</strong> {anime.episodes} | <strong>타입:</strong> {anime.type}
            </p>
            {/* 태그 섹션 (flex wrap 사용) */}
            <div className="mb-3">
                <strong className="text-[#333] block mb-1">장르:</strong>
                <div className="flex flex-wrap gap-1">
                    {anime.genres_list?.map((g, i) => <span key={`g-${i}`} className="inline-block bg-[#7FDBFF] text-[#333] px-2 py-1 rounded text-xs">{g}</span>)}
                </div>
            </div>
            <div className="mb-3">
                <strong className="text-[#333] block mb-1">스튜디오:</strong>
                 <div className="flex flex-wrap gap-1">
                    {anime.studios_list?.map((s, i) => <span key={`s-${i}`} className="inline-block bg-[#7FDBFF] text-[#333] px-2 py-1 rounded text-xs">{s}</span>)}
                 </div>
            </div>
            <div>
                <strong className="text-[#333] block mb-1">주요 태그 (일부):</strong>
                 <div className="flex flex-wrap gap-1">
                    {anime.tags_list?.slice(0, 15).map((t, i) => <span key={`t-${i}`} className="inline-block bg-[#7FDBFF] text-[#333] px-2 py-1 rounded text-xs">{t}</span>)}
                 </div>
            </div>
        </div>
    );
}

// 4. 추천 목록
function RecommendationList({ recommendations, onSelect, mainTitle }) {
    return (
        <div className="bg-transparent">
            <h3 className="text-xl font-semibold text-[#00A3D9] mb-4">
                {mainTitle ? `'${mainTitle}' 기반 추천` : '추천 목록'}
            </h3>
            <ul className="list-none p-0 space-y-2"> {/* 아이템 간 간격 */}
                {recommendations.map((rec, index) => (
                    <li
                        key={index}
                        className="bg-white p-4 rounded-lg shadow-md cursor-pointer transition duration-200 border-l-4 border-[#B2EBF2] hover:bg-[#E0F7FA] hover:border-[#00A3D9] group" /* group 추가 */
                        onClick={() => onSelect(rec.title)}
                    >
                        <div className="text-base font-semibold text-[#333] group-hover:text-[#00A3D9] transition duration-200">{rec.title}</div> {/* 호버 시 색상 변경 */}
                        <div className="text-xs text-[#555] mt-1"> {/* 폰트 크기 조정 */}
                            {rec.similarity_score === "N/A" ? (
                                <span className="font-bold text-[#00A3D9]">{rec.common_genres?.join(', ')}</span>
                            ) : (
                                <>
                                    유사도: <span className="font-bold text-[#00A3D9]">{rec.similarity_score}점</span>
                                    {rec.common_genres?.length > 0 && ` | 공통 장르: ${rec.common_genres.join(', ')}`}
                                </>
                            )}
                        </div>
                    </li>
                ))}
                {recommendations.length === 0 && mainTitle && (
                     <p className="text-[#555] text-center mt-4">추천 목록을 생성 중이거나, 추천할 항목이 없습니다.</p>
                )}
            </ul>
        </div>
    );
}


// --- 메인 App 컴포넌트 ---
function App() {
    const [selectedAnime, setSelectedAnime] = useState(null);
    const [recommendations, setRecommendations] = useState([]);
    const [mode, setMode] = useState('cbf');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [lastQuery, setLastQuery] = useState('');

    const handleSearch = useCallback(async (query) => {
        if (!query) return;
        setLoading(true);
        setError(null);
        setSelectedAnime(null);
        setRecommendations([]);
        setLastQuery(query);

        try {
            const response = await fetch(`${API_BASE_URL}/recommend?title=${query}&mode=${mode}`);
            const data = await response.json();
            if (response.ok && !data.error) {
                setSelectedAnime(data.main_anime);
                setRecommendations(data.recommendations);
            } else { setError(data.error || '추천 정보를 가져오는데 실패했습니다.'); }
        } catch (err) {
            console.error("Search fetch error:", err);
            setError('서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인하세요.');
        } finally { setLoading(false); }
    }, [mode]);

    return (
        // Tailwind 색상 직접 사용 (예: bg-[#E0F7FA])
        <div className="p-5 md:p-8 bg-[#E0F7FA] min-h-screen"> {/* 패딩 조정 */}
            {/* 제목 */}
            <h1 className="text-3xl font-bold border-b-2 border-[#B2EBF2] pb-3 mb-6 text-[#00A3D9] text-center"> {/* 가운데 정렬 */}
                애니 추천시스템
            </h1>

            {/* 검색창 */}
            <SearchBar onSearch={handleSearch} />
            {/* 모드 선택 */}
            <ModeSelector selectedMode={mode} onChange={setMode} />

            <hr className="border border-[#B2EBF2] mt-6" />

            {/* 로딩/에러 */}
            {loading && <p className="text-center text-lg mt-6 text-[#00A3D9]">로딩 중...</p>}
            {error && <p className="text-center text-red-600 text-lg mt-6">오류: {error}</p>}

            {/* 결과 영역 */}
            <div className="flex flex-col lg:flex-row gap-8 mt-6"> {/* 반응형 레이아웃 (lg 기준) */}
                <div className="lg:w-2/3"> {/* 메인 영역 */}
                    <AnimeInfoCard anime={selectedAnime} />
                </div>
                <div className="lg:w-1/3"> {/* 사이드바 영역 */}
                    <RecommendationList
                        recommendations={recommendations}
                        onSelect={handleSearch}
                        mainTitle={selectedAnime?.title}
                    />
                </div>
            </div>
        </div>
    );
}

export default App;

