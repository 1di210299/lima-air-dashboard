import React, { useState, useEffect } from 'react';
import { UserProfile, RunningRisk, Station } from '../types';
import { calculateRunningRisk, getRiskColor } from '../utils/airQuality';
import { Activity, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

interface RunningRiskWidgetProps {
  stations: Station[];
  selectedStation: string | null;
}

const RunningRiskWidget: React.FC<RunningRiskWidgetProps> = ({
  stations,
  selectedStation
}) => {
  const [userProfile, setUserProfile] = useState<UserProfile>({
    age: 30,
    healthCondition: 'none',
    preferredDistrict: selectedStation || '',
    notificationsEnabled: false
  });
  
  const [risk, setRisk] = useState<RunningRisk | null>(null);

  useEffect(() => {
    if (selectedStation && userProfile) {
      const station = stations.find(s => s.name === selectedStation);
      if (station) {
        // Simular datos actuales de PM2.5 basados en AQI
        const estimatedPM25 = station.currentAQI * 0.4; // Aproximación
        const estimatedPM10 = station.currentAQI * 0.8;
        
        const calculatedRisk = calculateRunningRisk(
          estimatedPM25,
          estimatedPM10,
          userProfile.age,
          userProfile.healthCondition
        );
        setRisk(calculatedRisk);
      }
    }
  }, [selectedStation, userProfile, stations]);

  const getRiskIcon = (level: string) => {
    switch (level) {
      case 'low':
        return <CheckCircle className="w-8 h-8 text-green-500" />;
      case 'medium':
        return <AlertTriangle className="w-8 h-8 text-yellow-500" />;
      case 'high':
        return <AlertTriangle className="w-8 h-8 text-orange-500" />;
      case 'extreme':
        return <XCircle className="w-8 h-8 text-red-500" />;
      default:
        return <Activity className="w-8 h-8 text-gray-500" />;
    }
  };

  const getRiskTitle = (level: string) => {
    switch (level) {
      case 'low': return 'Riesgo Bajo';
      case 'medium': return 'Riesgo Moderado';
      case 'high': return 'Riesgo Alto';
      case 'extreme': return 'Riesgo Extremo';
      default: return 'Sin datos';
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <Activity className="w-6 h-6 text-blue-500" />
        <h2 className="text-xl font-semibold">Riesgo para Correr</h2>
      </div>

      {/* User Profile Form */}
      <div className="mb-6 space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Edad
          </label>
          <input
            type="number"
            min="1"
            max="100"
            value={userProfile.age}
            onChange={(e) => setUserProfile(prev => ({
              ...prev,
              age: parseInt(e.target.value) || 30
            }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Condición de Salud
          </label>
          <select
            value={userProfile.healthCondition}
            onChange={(e) => setUserProfile(prev => ({
              ...prev,
              healthCondition: e.target.value as UserProfile['healthCondition']
            }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="none">Sin condiciones</option>
            <option value="asthma">Asma</option>
            <option value="respiratory">Problemas respiratorios</option>
            <option value="heart">Problemas cardíacos</option>
          </select>
        </div>
      </div>

      {/* Risk Assessment */}
      {risk && selectedStation ? (
        <div className="space-y-4">
          <div 
            className="p-4 rounded-lg border-2"
            style={{ 
              borderColor: getRiskColor(risk.level),
              backgroundColor: getRiskColor(risk.level) + '10'
            }}
          >
            <div className="flex items-center gap-3 mb-3">
              {getRiskIcon(risk.level)}
              <div>
                <h3 className="text-lg font-semibold">
                  {getRiskTitle(risk.level)}
                </h3>
                <p className="text-sm text-gray-600">
                  Puntuación: {risk.score}/140
                </p>
              </div>
            </div>
            
            <p className="text-sm text-gray-700 mb-4">
              {risk.recommendation}
            </p>

            {/* Risk Factors Breakdown */}
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-700">
                Factores de riesgo:
              </h4>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="text-center p-2 bg-gray-50 rounded">
                  <div className="font-medium">Calidad del aire</div>
                  <div className="text-gray-600">{risk.factors.airQuality}/100</div>
                </div>
                <div className="text-center p-2 bg-gray-50 rounded">
                  <div className="font-medium">Edad</div>
                  <div className="text-gray-600">{risk.factors.age}/30</div>
                </div>
                <div className="text-center p-2 bg-gray-50 rounded">
                  <div className="font-medium">Salud</div>
                  <div className="text-gray-600">{risk.factors.health}/20</div>
                </div>
              </div>
            </div>
          </div>

          {/* Station Info */}
          <div className="text-sm text-gray-600">
            <p>📍 Estación: {selectedStation}</p>
            <p>🕐 Última actualización: {new Date().toLocaleTimeString('es-PE')}</p>
          </div>
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          <Activity className="w-12 h-12 mx-auto mb-2 opacity-50" />
          <p>Selecciona una estación en el mapa para ver el riesgo</p>
        </div>
      )}
    </div>
  );
};

export default RunningRiskWidget;
