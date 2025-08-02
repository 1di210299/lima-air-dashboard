import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import { Station } from '../types';
import 'leaflet/dist/leaflet.css';

interface AirQualityMapProps {
  stations: Station[];
  selectedStation: string | null;
  onStationSelect: (stationName: string) => void;
}

// Component to fit map bounds
const FitBounds: React.FC<{ stations: Station[] }> = ({ stations }) => {
  const map = useMap();
  
  useEffect(() => {
    if (stations.length > 0) {
      const bounds = stations.map(station => station.coordinates);
      map.fitBounds(bounds, { padding: [20, 20] });
    }
  }, [stations, map]);
  
  return null;
};

const AirQualityMap: React.FC<AirQualityMapProps> = ({ 
  stations, 
  selectedStation, 
  onStationSelect 
}) => {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(false);
  }, [stations]);

  const getMarkerSize = (aqi: number): number => {
    if (aqi <= 50) return 15;
    if (aqi <= 100) return 20;
    if (aqi <= 150) return 25;
    if (aqi <= 200) return 30;
    return 35;
  };

  const center: [number, number] = [-12.0464, -77.0428]; // Lima center

  if (loading) {
    return (
      <div className="h-96 bg-gray-100 rounded-lg flex items-center justify-center">
        <div className="text-gray-500">Cargando mapa...</div>
      </div>
    );
  }

  return (
    <div className="h-96 rounded-lg overflow-hidden border">
      <MapContainer
        center={center}
        zoom={11}
        style={{ height: '100%', width: '100%' }}
        className="z-0"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        <FitBounds stations={stations} />
        
        {stations.map((station) => (
          <CircleMarker
            key={station.name}
            center={station.coordinates}
            radius={getMarkerSize(station.currentAQI)}
            fillColor={station.status.color}
            color="white"
            weight={2}
            opacity={1}
            fillOpacity={0.8}
            eventHandlers={{
              click: () => onStationSelect(station.name),
            }}
          >
            <Popup>
              <div className="p-2">
                <h3 className="font-bold text-lg">{station.name}</h3>
                <p className="text-sm text-gray-600">{station.distrito}</p>
                <div className="mt-2">
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: station.status.color }}
                    ></div>
                    <span className="font-medium">AQI: {station.currentAQI}</span>
                  </div>
                  <p className="text-sm mt-1">{station.status.description}</p>
                </div>
                <button
                  onClick={() => onStationSelect(station.name)}
                  className="mt-2 px-3 py-1 bg-blue-500 text-white text-sm rounded hover:bg-blue-600"
                >
                  Ver detalles
                </button>
              </div>
            </Popup>
          </CircleMarker>
        ))}
      </MapContainer>
    </div>
  );
};

export default AirQualityMap;
