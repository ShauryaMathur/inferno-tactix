// MapView.tsx
import React from 'react';
import { MapContainer, Marker, TileLayer, useMap, useMapEvents } from 'react-leaflet';

interface Props {
  center: [number, number];
  zoom: number;
  markerPos: [number, number] | null;
  theme: string;
  onClick: (pos: [number, number]) => void;
}

function Recenter({ center }: { center: [number, number] }) {
  const map = useMap();
  map.setView(center);
  return null;
}

function ClickableMarker({ onClick }: { onClick: (pos: [number, number]) => void }) {
  useMapEvents({
    click(e) {
      const newPos: [number, number] = [e.latlng.lat, e.latlng.lng];
      onClick(newPos);
    }
  });

  return null;
}

export const MapView: React.FC<Props> = ({ center, zoom, markerPos, theme, onClick }) => {
  const USA_BOUNDS: [[number, number], [number, number]] = [
    [24.396308, -124.848974],
    [49.384358, -66.885444]
  ];

  return (
    <MapContainer
      center={center}
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
      <Recenter center={center} />
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
      <ClickableMarker onClick={onClick} />
      {markerPos && <Marker position={markerPos} />}
    </MapContainer>
  );
};