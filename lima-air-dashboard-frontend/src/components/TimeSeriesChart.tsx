import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { TimeSeriesData, ForecastData } from '../types';
import { formatDateTime } from '../utils/airQuality';

interface TimeSeriesChartProps {
  historicalData: TimeSeriesData[];
  forecastData: ForecastData[];
  station: string;
}

const TimeSeriesChart: React.FC<TimeSeriesChartProps> = ({
  historicalData,
  forecastData,
  station
}) => {
  // Combinar datos históricos y de pronóstico
  const combinedData = [
    ...historicalData.map(item => ({
      timestamp: new Date(item.timestamp).getTime(),
      pm25_actual: item.pm2_5,
      pm10_actual: item.pm10,
      no2_actual: item.no2,
      type: 'historical'
    })),
    ...forecastData.slice(0, 24).map(item => ({ // Solo próximas 24 horas
      timestamp: new Date(item.timestamp).getTime(),
      pm25_forecast: item.predicted_pm2_5,
      pm10_forecast: item.predicted_pm10,
      no2_forecast: item.predicted_no2,
      type: 'forecast'
    }))
  ].sort((a, b) => a.timestamp - b.timestamp);

  const formatXAxis = (tickItem: number) => {
    return new Date(tickItem).toLocaleDateString('es-PE', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit'
    });
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border rounded-lg shadow-lg">
          <p className="font-semibold">
            {formatDateTime(new Date(label).toISOString())}
          </p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.name}: {entry.value?.toFixed(1)} µg/m³
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full">
      <h3 className="text-lg font-semibold mb-4">
        Histórico y Pronóstico - {station}
      </h3>
      
      {/* PM2.5 Chart */}
      <div className="mb-8">
        <h4 className="text-md font-medium mb-2 text-gray-700">PM2.5 (µg/m³)</h4>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={combinedData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="timestamp"
              type="number"
              scale="time"
              domain={['dataMin', 'dataMax']}
              tickFormatter={formatXAxis}
            />
            <YAxis />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line
              type="monotone"
              dataKey="pm25_actual"
              stroke="#2563eb"
              strokeWidth={2}
              dot={false}
              name="PM2.5 Real"
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="pm25_forecast"
              stroke="#dc2626"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="PM2.5 Pronóstico"
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* PM10 Chart */}
      <div className="mb-8">
        <h4 className="text-md font-medium mb-2 text-gray-700">PM10 (µg/m³)</h4>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={combinedData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="timestamp"
              type="number"
              scale="time"
              domain={['dataMin', 'dataMax']}
              tickFormatter={formatXAxis}
            />
            <YAxis />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line
              type="monotone"
              dataKey="pm10_actual"
              stroke="#059669"
              strokeWidth={2}
              dot={false}
              name="PM10 Real"
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="pm10_forecast"
              stroke="#dc2626"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="PM10 Pronóstico"
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* NO2 Chart */}
      <div>
        <h4 className="text-md font-medium mb-2 text-gray-700">NO2 (µg/m³)</h4>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={combinedData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="timestamp"
              type="number"
              scale="time"
              domain={['dataMin', 'dataMax']}
              tickFormatter={formatXAxis}
            />
            <YAxis />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line
              type="monotone"
              dataKey="no2_actual"
              stroke="#7c3aed"
              strokeWidth={2}
              dot={false}
              name="NO2 Real"
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="no2_forecast"
              stroke="#dc2626"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="NO2 Pronóstico"
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default TimeSeriesChart;
