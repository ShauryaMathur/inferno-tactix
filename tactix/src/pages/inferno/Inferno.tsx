// Inferno.tsx
import axios from 'axios';
import 'leaflet/dist/leaflet.css';
import React, { useState, useRef, useEffect } from 'react';
import styles from './inferno.module.scss';
import { LocationSearch } from './LocationSearch';
import { DatePicker } from './DatePicker';
import { MapView } from './MapView';
import { PredictionPanel } from './PredictionPanel';
import { LoadingSpinner } from './LoadingSpinner';

const DEFAULT_CENTER: [number, number] = [24.396308, -124.848974];
const DEFAULT_ZOOM = 4;
const MIN_DATE = '1980-01-01';
const MAX_DATE = new Date().toISOString().slice(0, 10);
const USA_BOUNDS: [[number, number], [number, number]] = [
  [24.396308, -124.848974],
  [49.384358, -66.885444]
];

interface ApiResponse {
  prediction: number;
}

export default function Inferno() {
  const [query, setQuery] = useState('');
  const [markerPos, setMarkerPos] = useState<[number, number] | null>(null);
  const [mapCenter, setMapCenter] = useState(DEFAULT_CENTER);
  const [zoom, setZoom] = useState(DEFAULT_ZOOM);
  const [date, setDate] = useState(MAX_DATE);
  const [theme] = useState('dark');
  const [apiResponse, setApiResponse] = useState<ApiResponse | null>(null);
  const [suggestions, setSuggestions] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [isCreatingEnvironment, setIsCreatingEnvironment] = useState(false);
  const [isFetchingPrediction, setIsFetchingPrediction] = useState(false);
  const searchTimeout = useRef<any>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (!(event.target as Element).closest('.search-container')) {
        setShowSuggestions(false);
      }
    }
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  const fetchSuggestions = async (input: string) => {
    if (!input || input.length < 2) {
      setSuggestions([]);
      return;
    }
    if (searchTimeout.current) clearTimeout(searchTimeout.current);
    searchTimeout.current = setTimeout(async () => {
      setIsLoading(true);
      try {
        const coordMatch = input.match(/^(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)$/);
        if (coordMatch) {
          setSuggestions([]);
          setShowSuggestions(false);
          return;
        }
        const resp = await fetch(`https://nominatim.openstreetmap.org/search?format=json&limit=5&q=${encodeURIComponent(input)}`);
        const results = await resp.json();
        setSuggestions(results);
        setShowSuggestions(true);
      } catch (err) {
        console.error('Suggestion fetch failed', err);
        setSuggestions([]);
      } finally {
        setIsLoading(false);
      }
    }, 300);
  };

  const performSearch = async () => {
    const trimmed = query.trim();
    if (!trimmed) return null;

    let lat: number, lon: number;
    const coordMatch = trimmed.match(/^(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)$/);
    if (coordMatch) {
      lat = parseFloat(coordMatch[1]);
      lon = parseFloat(coordMatch[3]);
    } else {
      try {
        const resp = await fetch(`https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(trimmed)}`);
        const results = await resp.json();
        if (!results.length) return alert('No results found');
        lat = +results[0].lat;
        lon = +results[0].lon;
      } catch (e) {
        console.error('Search error', e);
        return;
      }
    }

    if (
      lat < USA_BOUNDS[0][0] ||
      lat > USA_BOUNDS[1][0] ||
      lon < USA_BOUNDS[0][1] ||
      lon > USA_BOUNDS[1][1]
    ) {
      alert('Out of bounds');
      return;
    }

    const newPos: [number, number] = [lat, lon];
    setMarkerPos(newPos);
    setMapCenter(newPos);
    setZoom(12);
    return newPos;
  };

  const handleSuggestionClick = (suggestion: any) => {
    setQuery(suggestion.display_name);
    setSuggestions([]);
    setShowSuggestions(false);

    const lat = parseFloat(suggestion.lat);
    const lon = parseFloat(suggestion.lon);

    if (
      lat < USA_BOUNDS[0][0] ||
      lat > USA_BOUNDS[1][0] ||
      lon < USA_BOUNDS[0][1] ||
      lon > USA_BOUNDS[1][1]
    ) {
      alert('Out of bounds');
      return;
    }

    const newPos: [number, number] = [lat, lon];
    setMarkerPos(newPos);
    setMapCenter(newPos);
    setZoom(12);
  };

  const handlePositionChange = (pos: [number, number]) => {
    setMarkerPos(pos);
    setMapCenter(pos);
    setZoom(12);
  };

  const callApi = async (position: [number, number], selectedDate: string) => {
    setIsFetchingPrediction(true);
    try {
      const response = await axios.post('http://localhost:6969/api/predictWildfire', null, {
        params: { lat: position[0], lon: position[1], date: selectedDate }
      });
      if (response.data.prediction) {
        setApiResponse(response.data);
      } else {
        setApiResponse(null);
        alert('Some Error Occurred!');
      }
    } catch (error) {
      console.error('Error calling API:', error);
    } finally {
      setIsFetchingPrediction(false);
    }
  };

  const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

  const createEnvironmentAndNavigate = async () => {
    if (!markerPos) return alert('Please select a location on the map first');

    setIsCreatingEnvironment(true);
    try {
      await axios.post(`http://localhost:6969/api/createEnvironment`, null, {
        params: { lat: markerPos[0], lon: markerPos[1], date }
      });
      const query = new URLSearchParams({ lat: markerPos[0].toString(), lon: markerPos[1].toString(), date }).toString();
    //   await sleep(7000);
      window.open(`/#/tactics?${query}`, '_blank');
    } catch (error) {
      console.error('Error creating environment:', error);
      alert('Failed to create environment. Please try again.');
    } finally {
      setIsCreatingEnvironment(false);
    }
  };

  const handleSearch = async () => {
    const pos = await performSearch();
    if (pos) {
      setApiResponse(null);
      callApi(pos, date);
    }
  };

  const handleInputChange = (val: string) => {
    setQuery(val);
    fetchSuggestions(val);
  };

  return (
    <div className="home-page" style={{ padding: '1rem' }}>
      {isCreatingEnvironment && <LoadingSpinner message="Creating environment… Please wait" />}
      {isFetchingPrediction && <LoadingSpinner message="Getting prediction…" />}

      <div className={styles.searchBar} style={{ maxWidth: '800px', margin: '1rem auto', display: 'flex', gap: '2.5rem', alignItems: 'center', position: 'relative' }}>
        <LocationSearch
          query={query}
          setQuery={setQuery}
          suggestions={suggestions}
          setSuggestions={setSuggestions}
          isLoading={isLoading}
          setIsLoading={setIsLoading}
          showSuggestions={showSuggestions}
          setShowSuggestions={setShowSuggestions}
          setMarkerPos={setMarkerPos}
          setMapCenter={setMapCenter}
          setZoom={setZoom}
          onSuggestionClick={handleSuggestionClick}
          onInputChange={handleInputChange}
          onKeyEnter={handleSearch}
        />
        <DatePicker value={date} onChange={setDate} min={MIN_DATE} max={MAX_DATE} />
        <button style={{ padding: '0.75rem 1.5rem', fontSize: '1rem', background: '#ffdf00c2', color: '#000', border: 'none', borderRadius: '4px', cursor: 'pointer' }} onClick={handleSearch}>Analyze Risk</button>
      </div>

      <div className={styles.preview}>
        <div className={styles.previewFrame} style={{ display: 'flex', justifyContent: 'center' }}>
          <MapView center={mapCenter} zoom={zoom} markerPos={markerPos} theme={theme} onClick={handlePositionChange} />
        </div>
      </div>

      {(
        <PredictionPanel
          prediction={0.8}
          date={date}
          isCreating={isCreatingEnvironment}
          onCreate={createEnvironmentAndNavigate}
          theme={theme}
        />
      )}
    </div>
  );
}
