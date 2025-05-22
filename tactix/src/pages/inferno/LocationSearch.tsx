// LocationSearch.tsx
import React from 'react';

export function LocationSearch({
  query,
  setQuery,
  suggestions,
  setSuggestions,
  isLoading,
  setIsLoading,
  showSuggestions,
  setShowSuggestions,
  setMarkerPos,
  setMapCenter,
  setZoom,
  onSuggestionClick,
  onInputChange,
  onKeyEnter
}: {
  query: string;
  setQuery: (q: string) => void;
  suggestions: any[];
  setSuggestions: (s: any[]) => void;
  isLoading: boolean;
  setIsLoading: (b: boolean) => void;
  showSuggestions: boolean;
  setShowSuggestions: (b: boolean) => void;
  setMarkerPos: (pos: [number, number]) => void;
  setMapCenter: (pos: [number, number]) => void;
  setZoom: (z: number) => void;
  onSuggestionClick: (s: any) => void;
  onInputChange: (val: string) => void;
  onKeyEnter: () => void;
}) {
  return (
    <div className="search-container" style={{ position: 'relative', flex: 1 }}>
      <input
        type="text"
        placeholder="City Name or lat,lon"
        value={query}
        onChange={(e) => onInputChange(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && onKeyEnter()}
        style={{ width: '100%' }}
      />

      {showSuggestions && suggestions.length > 0 && (
        <ul
          style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            width: '100%',
            maxHeight: '200px',
            overflowY: 'auto',
            margin: 0,
            padding: 0,
            listStyle: 'none',
            background: '#333',
            border: '1px solid #444',
            borderRadius: '4px',
            zIndex: 1000
          }}
        >
          {suggestions.map((suggestion, index) => (
            <li
              key={index}
              onClick={() => onSuggestionClick(suggestion)}
              onMouseDown={(e) => e.preventDefault()}
              style={{
                padding: '8px 12px',
                cursor: 'pointer',
                borderBottom: '1px solid #444',
                color: '#fff'
              }}
            >
              {suggestion.display_name}
            </li>
          ))}
        </ul>
      )}

      {isLoading && (
        <div style={{
          position: 'absolute',
          right: '10px',
          top: '50%',
          transform: 'translateY(-50%)',
          color: '#999'
        }}>
          Loading...
        </div>
      )}
    </div>
  );
}
