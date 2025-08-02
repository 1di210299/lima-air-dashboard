export interface AirQualityData {
  id: number;
  estacion: string;
  fecha: string;
  hora: string;
  longitud: number;
  latitud: number;
  altitud: number;
  pm10: number | null;
  pm2_5: number | null;
  no2: number | null;
  departamento: string;
  provincia: string;
  distrito: string;
  ubigeo: string;
  fecha_corte: string;
}

export interface Station {
  name: string;
  distrito: string;
  coordinates: [number, number];
  currentAQI: number;
  status: AirQualityStatus;
}

export interface AirQualityStatus {
  level: 'good' | 'moderate' | 'unhealthy' | 'dangerous' | 'hazardous';
  color: string;
  description: string;
}

export interface TimeSeriesData {
  timestamp: string;
  pm10: number | null;
  pm2_5: number | null;
  no2: number | null;
  station: string;
}

export interface ForecastData {
  timestamp: string;
  predicted_pm10: number;
  predicted_pm2_5: number;
  predicted_no2: number;
  station: string;
  confidence: number;
}

export interface RunningRisk {
  level: 'low' | 'medium' | 'high' | 'extreme';
  score: number;
  recommendation: string;
  factors: {
    airQuality: number;
    age: number;
    health: number;
  };
}

export interface UserProfile {
  age: number;
  healthCondition: 'none' | 'asthma' | 'heart' | 'respiratory';
  preferredDistrict: string;
  notificationsEnabled: boolean;
}
