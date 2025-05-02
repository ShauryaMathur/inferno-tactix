import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, useMap, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import styles from './inferno.module.scss';
import axios from 'axios'; // Import axios for API calls

const DEFAULT_CENTER: [number, number] = [24.396308, -124.848974];
const DEFAULT_ZOOM = 4;

// Bounding box for the lower 48 states:
const USA_BOUNDS: [[number, number], [number, number]] = [
    [24.396308, -124.848974],  // SW corner
    [49.384358, -66.885444]   // NE corner
];
const MIN_DATE = '1980-01-01';
const now = new Date();
const maxDate = new Date(now.getTime());
const MAX_DATE = maxDate.toISOString().slice(0, 10); // Fixed to get only the date part

function Recenter({ center }: { center: [number, number] }) {
    const map = useMap();
    map.setView(center);
    return null;
}

// Modified to only update position without API call
function ClickableMarker({ onPositionChange }: { onPositionChange: (pos: [number, number]) => void }) {
    const [pos, setPos] = useState<[number, number] | null>(null);

    useMapEvents({
        click(e) {
            const newPos: [number, number] = [e.latlng.lat, e.latlng.lng];
            setPos(newPos);
            onPositionChange(newPos);
        }
    });

    return pos ? <Marker position={pos} /> : null;
}

export default function Inferno() {
    const [query, setQuery] = useState('');
    const [markerPos, setMarkerPos] = useState<[number, number] | null>(null);
    const [mapCenter, setMapCenter] = useState<[number, number]>(DEFAULT_CENTER);
    const [zoom, setZoom] = useState(DEFAULT_ZOOM);
    const [date, setDate] = useState(now.toISOString().slice(0, 10)); // Store only the date
    const [theme, setTheme] = useState('dark');
    const [apiResponse, setApiResponse] = useState(null);
    // New state for autocomplete
    const [suggestions, setSuggestions] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [showSuggestions, setShowSuggestions] = useState(false);
    const searchTimeout = useRef<any>(null);

    // Function to fetch location suggestions with debouncing
    const fetchSuggestions = async (input: string) => {
        console.log(input);
        
        if (!input || input.length < 2) {
            setSuggestions([]);
            return;
        }
        
        // Clear any existing timeout
        if (searchTimeout.current) {
            clearTimeout(searchTimeout.current);
        }
        
        // Set a new timeout for debouncing
        searchTimeout.current = setTimeout(async () => {
            setIsLoading(true);
            try {
                const coordMatch = input.match(/^(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)$/);
                if (coordMatch) {
                    // Direct coordinate input, no need for suggestions
                    setSuggestions([]);
                    setShowSuggestions(false);
                    return;
                }
                
                const resp = await fetch(
                    `https://nominatim.openstreetmap.org/search?format=json&limit=5&q=${encodeURIComponent(input)}`
                );
                const results = await resp.json();
                setSuggestions(results);
                setShowSuggestions(true);
            } catch (error) {
                console.error('Error fetching suggestions:', error);
                setSuggestions([]);
            } finally {
                setIsLoading(false);
            }
        }, 300); // 300ms debounce time
    };

    // Handle input change
    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = e.target.value;
        setQuery(value);
        fetchSuggestions(value);
    };

    // Handle suggestion click
    const handleSuggestionClick = (suggestion: any) => {
        setQuery(suggestion.display_name);
        setSuggestions([]);
        setShowSuggestions(false);
        
        const lat = parseFloat(suggestion.lat);
        const lon = parseFloat(suggestion.lon);
        
        // optional: bound check
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

    // Click outside handler to close the suggestion dropdown
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (!(event.target as Element).closest('.search-container')) {
                setShowSuggestions(false);
            }
        }
        
        document.addEventListener('click', handleClickOutside);
        return () => {
            document.removeEventListener('click', handleClickOutside);
        };
    }, []);

    // Handle marker position changes from map clicks - just update UI, no API call
    const handlePositionChange = (pos: [number, number]) => {
        setMarkerPos(pos);
        setMapCenter(pos);
        setZoom(12);

        // No API call here anymore
    };

    // Call backend API
    const callApi = async (position: [number, number], selectedDate: string) => {
        try {
            const response = await axios.post('/api/your-endpoint', {
                latitude: position[0],
                longitude: position[1],
                date: selectedDate
            });

            setApiResponse(response.data);
            console.log('API Response:', response.data);
        } catch (error) {
            console.error('Error calling API:', error);
        }
    };

    // Handle date change
    const handleDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newDate = e.target.value;
        setDate(newDate);

        // No automatic API call when date changes
    };

    // Split into two functions: one for searching only and one for searching + API call
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
                const resp = await fetch(
                    `https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(trimmed)}`
                );
                const results = await resp.json();
                console.log(results);

                if (!results.length) {
                    return alert('No results found');
                }
                lat = +results[0].lat;
                lon = +results[0].lon;
                console.log('Lat:', lat, 'Lon:', lon);

                // optional: bound check
                if (
                    lat < USA_BOUNDS[0][0] ||
                    lat > USA_BOUNDS[1][0] ||
                    lon < USA_BOUNDS[0][1] ||
                    lon > USA_BOUNDS[1][1]
                ) {
                    return alert('Out of bounds');
                }

                setMarkerPos([lat, lon]);
                setMapCenter([lat, lon]);
                setZoom(12);
            } catch (error) {
                console.error('Error searching for location:', error);
                alert('Error searching for location');
                return null;
            }
        }

        // optional: bound check
        if (
            lat < USA_BOUNDS[0][0] ||
            lat > USA_BOUNDS[1][0] ||
            lon < USA_BOUNDS[0][1] ||
            lon > USA_BOUNDS[1][1]
        ) {
            alert('Out of bounds');
            return null;
        }

        const newPos: [number, number] = [lat, lon];
        setMarkerPos(newPos);
        setMapCenter(newPos);
        setZoom(12);

        return newPos;
    };

    // This function handles search and calls the API - only triggered by Go button
    const handleSearch = async () => {
        const newPos = await performSearch();
        if (newPos) {
            // Only call API when using the Go button
            callApi(newPos, date);
        }
    };

    return (
        <div className="home-page" style={{ padding: '1rem' }}>
            {/* search + date */}
            <div className={styles.searchBar} style={{ maxWidth: '800px', margin: '1rem auto', display: 'flex', gap: '2.5rem', alignItems: 'center', position: 'relative' }}>
                <div className="search-container" style={{ position: 'relative', flex: 1 }}>
                    <input 
                        type="text" 
                        placeholder="City Name or lat,lon" 
                        value={query} 
                        onChange={handleInputChange}
                        onKeyDown={e => e.key === 'Enter' && performSearch()} 
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
                                background: theme === 'dark' ? '#333' : '#fff',
                                border: theme === 'dark' ? '1px solid #444' : '1px solid #ddd',
                                borderRadius: '4px', 
                                zIndex: 1000,
                                boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                            }}
                        >
                            {suggestions.map((suggestion, index) => (
                                <li 
                                    key={index}
                                    onClick={() => handleSuggestionClick(suggestion)}
                                    onMouseDown={(e) => e.preventDefault()} // Prevent blur event from hiding dropdown
                                    style={{ 
                                        padding: '8px 12px',
                                        cursor: 'pointer',
                                        borderBottom: theme === 'dark' ? '1px solid #444' : '1px solid #eee',
                                        color: theme === 'dark' ? '#fff' : '#333',
                                        whiteSpace: 'nowrap',
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis'
                                    }}
                                    onMouseOver={(e) => {
                                        e.currentTarget.style.background = theme === 'dark' ? '#444' : '#f5f5f5';
                                    }}
                                    onMouseOut={(e) => {
                                        e.currentTarget.style.background = 'transparent';
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
                <input type="date" value={date} onChange={handleDateChange} min={MIN_DATE} max={MAX_DATE} /> 
                <button style={{ padding: '0.75rem 1.5rem', fontSize: '1rem', background: '#ffdf00c2', color: '#000', border: 'none', borderRadius: '4px', cursor: 'pointer' }} onClick={handleSearch} > Search & Get Data </button> 
                {/* <button style={{ padding: '0.75rem 1.5rem', fontSize: '1rem', background: theme === 'dark' ? '#fff' : '#333', color: theme === 'dark' ? '#000' : '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer' }} onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>{theme === 'dark' ? 'Light' : 'Dark'} Mode</button> */}
            </div>

            {/* centered map */}
            <div className={styles.preview}>
                <div className={styles.previewFrame} style={{ display: 'flex', justifyContent: 'center' }}>
                    <MapContainer
                        center={mapCenter}
                        zoom={zoom}
                        maxBounds={USA_BOUNDS}
                        maxBoundsViscosity={1}
                        style={{
                            width: '80vw',
                            height: '65vh',
                            margin: '0 auto',
                            borderRadius: '8px',
                            boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                        }}
                    >
                        <Recenter center={mapCenter} />
                        {theme === 'dark' ? (
                            <TileLayer
                                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                                attribution="&copy; <a href='https://carto.com/attributions'>CARTO</a>"
                            />
                        ) : (
                            <TileLayer
                                attribution="© OpenStreetMap contributors"
                                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                            />
                        )}
                        <ClickableMarker onPositionChange={handlePositionChange} />
                        {markerPos && <Marker position={markerPos} />}

                    </MapContainer>
                </div>
            </div>

            {/* Display API response if available */}
            {apiResponse && (
                <div style={{
                    maxWidth: '800px',
                    margin: '1rem auto',
                    padding: '1rem',
                    background: theme === 'dark' ? '#333' : '#f5f5f5',
                    color: theme === 'dark' ? '#fff' : '#000',
                    borderRadius: '8px'
                }}>
                    <h3>API Response</h3>
                    <pre>{JSON.stringify(apiResponse, null, 2)}</pre>
                </div>
            )}
        </div>
    );
}