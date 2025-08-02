import { AirQualityStatus, RunningRisk } from '../types';

export const getAirQualityStatus = (pm25: number | null, pm10: number | null, no2: number | null): AirQualityStatus => {
  // WHO 2021 guidelines for PM2.5: 
  // Good: 0-15, Moderate: 16-35, Unhealthy: 36-75, Dangerous: 76-150, Hazardous: >150
  
  const pm25Value = pm25 || 0;
  const pm10Value = pm10 || 0;
  
  if (pm25Value <= 15 && pm10Value <= 45) {
    return {
      level: 'good',
      color: '#22c55e',
      description: 'Buena - Aire limpio y saludable'
    };
  } else if (pm25Value <= 35 && pm10Value <= 90) {
    return {
      level: 'moderate',
      color: '#eab308',
      description: 'Moderada - Aceptable para la mayoría'
    };
  } else if (pm25Value <= 75 && pm10Value <= 180) {
    return {
      level: 'unhealthy',
      color: '#f97316',
      description: 'Dañina - Grupos sensibles en riesgo'
    };
  } else if (pm25Value <= 150 && pm10Value <= 360) {
    return {
      level: 'dangerous',
      color: '#ef4444',
      description: 'Peligrosa - Todos en riesgo'
    };
  } else {
    return {
      level: 'hazardous',
      color: '#7c2d12',
      description: 'Peligrosa - Emergencia sanitaria'
    };
  }
};

export const calculateRunningRisk = (
  pm25: number | null,
  pm10: number | null,
  age: number,
  healthCondition: string
): RunningRisk => {
  const pm25Value = pm25 || 0;
  const pm10Value = pm10 || 0;
  
  // Base score from air quality (0-100 scale)
  let airQualityScore = 0;
  if (pm25Value <= 15) airQualityScore = 20;
  else if (pm25Value <= 35) airQualityScore = 40;
  else if (pm25Value <= 75) airQualityScore = 60;
  else if (pm25Value <= 150) airQualityScore = 80;
  else airQualityScore = 100;
  
  // Age factor (0-30 scale)
  let ageScore = 0;
  if (age < 18) ageScore = 15;
  else if (age < 35) ageScore = 5;
  else if (age < 55) ageScore = 10;
  else if (age < 70) ageScore = 20;
  else ageScore = 25;
  
  // Health condition factor (0-20 scale)
  let healthScore = 0;
  switch (healthCondition) {
    case 'none': healthScore = 0; break;
    case 'asthma': healthScore = 20; break;
    case 'respiratory': healthScore = 15; break;
    case 'heart': healthScore = 10; break;
  }
  
  const totalScore = airQualityScore + ageScore + healthScore;
  
  if (totalScore <= 30) {
    return {
      level: 'low',
      score: totalScore,
      recommendation: '¡Excelente momento para correr! Las condiciones son favorables.',
      factors: { airQuality: airQualityScore, age: ageScore, health: healthScore }
    };
  } else if (totalScore <= 60) {
    return {
      level: 'medium',
      score: totalScore,
      recommendation: 'Puedes correr con precaución. Considera acortar la rutina.',
      factors: { airQuality: airQualityScore, age: ageScore, health: healthScore }
    };
  } else if (totalScore <= 90) {
    return {
      level: 'high',
      score: totalScore,
      recommendation: 'No recomendado correr al aire libre. Considera ejercicio en interiores.',
      factors: { airQuality: airQualityScore, age: ageScore, health: healthScore }
    };
  } else {
    return {
      level: 'extreme',
      score: totalScore,
      recommendation: 'Evita cualquier actividad física al aire libre. Mantente en interiores.',
      factors: { airQuality: airQualityScore, age: ageScore, health: healthScore }
    };
  }
};

export const formatDateTime = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleDateString('es-PE', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
};

export const getRiskColor = (level: string): string => {
  switch (level) {
    case 'low': return '#22c55e';
    case 'medium': return '#eab308';
    case 'high': return '#f97316';
    case 'extreme': return '#ef4444';
    default: return '#6b7280';
  }
};

export const getDistrictCoordinates = (distrito: string): [number, number] => {
  // Coordenadas aproximadas de los distritos principales de Lima
  const coordinates: { [key: string]: [number, number] } = {
    'JESUS_MARIA': [-12.0705, -77.0432],
    'SAN_MARTIN_DE_PORRES': [-11.9866, -77.0711],
    'ATE': [-12.0464, -76.9178],
    'HUACHIPA': [-12.0464, -76.9178],
    'SANTA_ANITA': [-12.0464, -76.9178],
    'LIMA': [-12.0464, -77.0428],
    'MIRAFLORES': [-12.1196, -77.0287],
    'SAN_ISIDRO': [-12.0988, -77.0378],
    'SURCO': [-12.1359, -77.0187],
    'LA_MOLINA': [-12.0867, -76.9378]
  };
  
  return coordinates[distrito] || [-12.0464, -77.0428]; // Lima centro por defecto
};
