import React, { useState, useEffect, useCallback } from 'react';

// --- API Base URL (Backend FastAPI Server) ---
const API_BASE_URL = 'http://127.0.0.1:5001';

// --- SVG Icon Components ---

// Minimal block-style logo icon
const SearchLogoIcon = () => (
    <svg width="28" height="28" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M20 80 V 20 H 40 V 50 H 60 V 20 H 80 V 80 H 60 V 50 H 40 V 80 H 20 Z" fill="#00A3D9"/>
    </svg>
);

// Magnifying glass icon
const SearchIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00A3D9" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8"></circle>
        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
    </svg>
);


// --- Components ---

// 1. Search bar with autocomplete suggestions
function SearchBar({ onSearch }) {
    const [query, setQuery] = useState('');
    const [suggestions, setSuggestions] = useState([]);
    const [debounceTimeout, setDebounceTimeout] = useState(null);

    const fetchSuggestions = useCallback(async (currentQuery) => {
        if (currentQuery.length < 1) { setSuggestions([]); return; }
        try {
            const response = await fetch(`${API_BASE_URL}/suggest?q=${currentQuery}`);
            if (response.ok) setSuggestions(await response.json());
            else setSuggestions([]);
        } catch (error) {
            console.error("Suggestion fetch error:", error);
            setSuggestions([]);
        }
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
        <div className="relative w-full max-w-xl mx-auto my-5">
            {/* Search Input */}
            <div className="flex items-center h-14 border-2 border-[#00A3D9] rounded-full bg-white pl-4 pr-2 shadow-sm">
                <div className="mr-3"><SearchLogoIcon /></div>
                <input
                    type="text" value={query} onChange={handleInputChange} onKeyPress={handleKeyPress}
                    placeholder="Enter anime title (Korean/English)..."
                    className="flex-grow border-none outline-none text-lg bg-transparent text-[#333]"
                    autoComplete="off"
                />
                <button
                    onClick={handleSearchClick}
                    className="h-10 w-10 flex items-center justify-center bg-transparent text-[#00A3D9] cursor-pointer rounded-full hover:bg-[#E0F7FA]"
                >
                    <SearchIcon />
                </button>
            </div>

            {/* Autocomplete Dropdown */}
            {suggestions.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-[#B2EBF2] rounded-lg shadow-lg z-10 max-h-60 overflow-y-auto">
                    {suggestions.map((title, index) => (
                        <div
                            key={index}
                            className="px-5 py-3 text-base cursor-pointer hover:bg-[#E0F7FA]"
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


// 2. Recommendation mode selector
function ModeSelector({ selectedMode, onChange }) {
    const modes = [
        { value: 'cbf', label: 'Content-Based' },
        { value: 'hybrid', label: 'Hybrid (Apriori)' },
        { value: 'lr_hybrid', label: 'Hybrid (Popularity Boost)' },
        { value: 'cf', label: 'Collaborative Filtering' },
    ];

    return (
        <div className="my-5 text-base text-[#333] text-center">
            <strong className="mr-4">Recommendation Mode:</strong>
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


// 3. Anime Information Card
function AnimeInfoCard({ anime }) {
    if (!anime) {
        return (
            <div className="bg-white rounded-lg p-6 shadow-lg min-h-[400px]">
                <h3 className="text-[#00A3D9] text-xl font-semibold">Anime Information</h3>
                <p className="text-[#555] mt-2">Search for an anime to display details here.</p>
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
                className="w-[225px] h-[318px] rounded-lg float-right ml-6 mb-4 border border-[#B2EBF2] object-cover shadow-md"
                onError={(e) => { e.target.src="https://via.placeholder.com/225x318.png?text=Image+Error" }}
            />

            <h2 className="text-2xl font-bold text-[#00A3D9] mt-0 mb-1">{anime.title}</h2>
            <p className="text-xs text-[#555] mb-3">(MAL ID: {anime.mal_id})</p>

            <p className="text-sm text-[#555] mb-3 clear-right">
                {japaneseName && `JP: ${japaneseName}`} {englishName && ` | EN: ${englishName}`}
            </p>

            <p className="text-base text-[#333] mb-4">
                <strong>Score:</strong> {anime.score?.toFixed(2)} | <strong>Episodes:</strong> {anime.episodes} | <strong>Type:</strong> {anime.type}
            </p>

            {/* Genre / Studio / Tags */}
            <div className="mb-3">
                <strong className="text-[#333] block mb-1">Genres:</strong>
                <div className="flex flex-wrap gap-1">
                    {anime.genres_list?.map((g, i) => <span key={i} className="bg-[#7FDBFF] px-2 py-1 rounded text-xs">{g}</span>)}
                </div>
            </div>

            <div className="mb-3">
                <strong className="text-[#333] block mb-1">Studios:</strong>
                <div className="flex flex-wrap gap-1">
                    {anime.studios_list?.map((s, i) => <span key={i} className="bg-[#7FDBFF] px-2 py-1 rounded text-xs">{s}</span>)}
                </div>
            </div>

            <div>
                <strong className="text-[#333] block mb-1">Tags (partial):</strong>
                <div className="flex flex-wrap gap-1">
                    {anime.tags_list?.slice(0, 15).map((t, i) => <span key={i} className="bg-[#7FDBFF] px-2 py-1 rounded text-xs">{t}</span>)}
                </div>
            </div>
        </div>
    );
}


// 4. Recommendation List
function RecommendationList({ recommendations, onSelect, mainTitle }) {
    return (
        <div className="bg-transparent">
            <h3 className="text-xl font-semibold text-[#00A3D9] mb-4">
                {mainTitle ? `Recommendations based on '${mainTitle}'` : 'Recommendations'}
            </h3>

            <ul className="list-none p-0 space-y-2">
                {recommendations.map((rec, index) => (
                    <li
                        key={index}
                        className="bg-white p-4 rounded-lg shadow-md cursor-pointer transition duration-200 border-l-4 border-[#B2EBF2] hover:bg-[#E0F7FA] hover:border-[#00A3D9] group"
                        onClick={() => onSelect(rec.title)}
                    >
                        <div className="text-base font-semibold text-[#333] group-hover:text-[#00A3D9]">
                            {rec.title}
                        </div>
                        <div className="text-xs text-[#555] mt-1">
                            {rec.similarity_score === "N/A" ? (
                                <span className="font-bold text-[#00A3D9]">{rec.common_genres?.join(', ')}</span>
                            ) : (
                                <>
                                    Similarity: <span className="font-bold text-[#00A3D9]">{rec.similarity_score}</span>
                                    {rec.common_genres?.length > 0 && ` | Shared Genres: ${rec.common_genres.join(', ')}`}
                                </>
                            )}
                        </div>
                    </li>
                ))}
                {recommendations.length === 0 && mainTitle && (
                     <p className="text-[#555] text-center mt-4">No recommendations available.</p>
                )}
            </ul>
        </div>
    );
}


// --- Main App Component ---
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
            } else setError(data.error || 'Failed to fetch recommendation data.');
        } catch (err) {
            console.error("Search fetch error:", err);
            setError('Cannot connect to backend. Ensure the server is running.');
        } finally { setLoading(false); }
    }, [mode]);

    return (
        <div className="p-5 md:p-8 bg-[#E0F7FA] min-h-screen">
            <h1 className="text-3xl font-bold border-b-2 border-[#B2EBF2] pb-3 mb-6 text-[#00A3D9] text-center">
                Anime Recommendation System
            </h1>

            <SearchBar onSearch={handleSearch} />
            <ModeSelector selectedMode={mode} onChange={setMode} />

            <hr className="border border-[#B2EBF2] mt-6" />

            {loading && <p className="text-center text-lg mt-6 text-[#00A3D9]">Loading...</p>}
            {error && <p className="text-center text-red-600 text-lg mt-6">Error: {error}</p>}

            <div className="flex flex-col lg:flex-row gap-8 mt-6">
                <div className="lg:w-2/3">
                    <AnimeInfoCard anime={selectedAnime} />
                </div>
                <div className="lg:w-1/3">
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