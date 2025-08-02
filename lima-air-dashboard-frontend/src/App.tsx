import React, { useState, useEffect } from 'react';
import { Station, TimeSeriesData, ForecastData } from './types';
import { dataService } from './services/dataService';
import AirQualityMap from './components/AirQualityMap';
import TimeSeriesChart from './components/TimeSeriesChart';
import RunningRiskWidget from './components/RunningRiskWidget';
import { Wind, MapPin, TrendingUp, Activity } from 'lucide-react';
import './App.css';

function App() {
  const [stations, setStations] = useState<Station[]>([]);
  const [selectedStation, setSelectedStation] = useState<string | null>(null);
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData[]>([]);
  const [forecastData, setForecastData] = useState<ForecastData[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'charts' | 'risk'>('overview');

  useEffect(() => {
    loadInitialData();
  }, []);

  useEffect(() => {
    if (selectedStation) {
      loadStationData(selectedStation);
    }
  }, [selectedStation]);

  const loadInitialData = async () => {
    try {
      const stationsData = await dataService.getCurrentStations();
      setStations(stationsData);
      
      if (stationsData.length > 0) {
        setSelectedStation(stationsData[0].name);
      }
    } catch (error) {
      console.error('Error loading initial data:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadStationData = async (stationName: string) => {
    try {
      const [timeSeries, forecast] = await Promise.all([
        dataService.getTimeSeriesData(stationName, 7),
        dataService.getForecastData(stationName)
      ]);
      
      setTimeSeriesData(timeSeries);
      setForecastData(forecast);
    } catch (error) {
      console.error('Error loading station data:', error);
    }
  };

  const handleStationSelect = (stationName: string) => {
    setSelectedStation(stationName);
  };

  const getOverallAirQuality = () => {
    if (stations.length === 0) return { level: 'unknown', color: '#6b7280', description: 'Sin datos' };
    
    const avgAQI = stations.reduce((sum, station) => sum + station.currentAQI, 0) / stations.length;
    
    if (avgAQI <= 50) return { level: 'good', color: '#22c55e', description: 'Buena' };
    if (avgAQI <= 100) return { level: 'moderate', color: '#eab308', description: 'Moderada' };
    if (avgAQI <= 150) return { level: 'unhealthy', color: '#f97316', description: 'Dañina' };
    if (avgAQI <= 200) return { level: 'dangerous', color: '#ef4444', description: 'Peligrosa' };
    return { level: 'hazardous', color: '#7c2d12', description: 'Peligrosa' };
  };

  const overallAQ = getOverallAirQuality();

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Wind className="w-12 h-12 text-blue-500 mx-auto mb-4 animate-spin" />
          <p className="text-gray-600">Cargando datos de calidad del aire...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <Wind className="w-8 h-8 text-blue-500" />
              <div>
                <h1 className="text-xl font-bold text-gray-900">
                  Lima Air Quality Dashboard
                </h1>
                <p className="text-sm text-gray-600">
                  Monitoreo en tiempo real de la calidad del aire
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="text-right">
                <div className="text-sm text-gray-600">Calidad general</div>
                <div 
                  className="text-lg font-semibold"
                  style={{ color: overallAQ.color }}
                >
                  {overallAQ.description}
                </div>
              </div>
              <div 
                className="w-4 h-4 rounded-full"
                style={{ backgroundColor: overallAQ.color }}
              ></div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            <button
              onClick={() => setActiveTab('overview')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'overview'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <div className="flex items-center gap-2">
                <MapPin className="w-4 h-4" />
                Vista General
              </div>
            </button>
            
            <button
              onClick={() => setActiveTab('charts')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'charts'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4" />
                Análisis Temporal
              </div>
            </button>
            
            <button
              onClick={() => setActiveTab('risk')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'risk'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4" />
                Riesgo Running
              </div>
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Map */}
            <div className="lg:col-span-2">
              <div className="bg-white p-6 rounded-lg shadow">
                <h2 className="text-lg font-semibold mb-4">
                  Mapa de Estaciones de Monitoreo
                </h2>
                <AirQualityMap
                  stations={stations}
                  selectedStation={selectedStation}
                  onStationSelect={handleStationSelect}
                />
              </div>
            </div>
            
            {/* Stations List */}
            <div className="space-y-4">
              <div className="bg-white p-6 rounded-lg shadow">
                <h2 className="text-lg font-semibold mb-4">Estaciones</h2>
                <div className="space-y-3">
                  {stations.map((station) => (
                    <div
                      key={station.name}
                      onClick={() => handleStationSelect(station.name)}
                      className={`p-3 rounded-lg border cursor-pointer transition-all ${
                        selectedStation === station.name
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium text-sm">
                            {station.name}
                          </div>
                          <div className="text-xs text-gray-600">
                            {station.distrito}
                          </div>
                        </div>
                        <div className="text-right">
                          <div 
                            className="text-sm font-semibold"
                            style={{ color: station.status.color }}
                          >
                            AQI {station.currentAQI}
                          </div>
                          <div 
                            className="w-3 h-3 rounded-full ml-auto"
                            style={{ backgroundColor: station.status.color }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'charts' && selectedStation && (
          <div className="bg-white p-6 rounded-lg shadow">
            <TimeSeriesChart
              historicalData={timeSeriesData}
              forecastData={forecastData}
              station={selectedStation}
            />
          </div>
        )}

        {activeTab === 'risk' && (
          <div className="max-w-md mx-auto">
            <RunningRiskWidget
              stations={stations}
              selectedStation={selectedStation}
            />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
