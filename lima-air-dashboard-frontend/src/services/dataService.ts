import { AirQualityData, TimeSeriesData, ForecastData, Station } from '../types';
import { getAirQualityStatus, getDistrictCoordinates } from '../utils/airQuality';

// Mock data service - En producción esto vendría de tu API backend
class DataService {
  private baseUrl = 'http://localhost:3001/api'; // Backend URL

  // Simular carga de datos desde CSV
  async loadAirQualityData(): Promise<AirQualityData[]> {
    // En producción, esto haría una llamada a tu API que procesa el CSV
    // Por ahora simulamos algunos datos
    return this.getMockData();
  }

  async getCurrentStations(): Promise<Station[]> {
    const data = await this.loadAirQualityData();
    const stationMap = new Map<string, AirQualityData[]>();
    
    // Agrupar por estación
    data.forEach(record => {
      if (!stationMap.has(record.estacion)) {
        stationMap.set(record.estacion, []);
      }
      stationMap.get(record.estacion)!.push(record);
    });

    // Obtener datos más recientes por estación
    const stations: Station[] = [];
    stationMap.forEach((records, stationName) => {
      const latest = records.sort((a, b) => 
        new Date(b.fecha + ' ' + b.hora).getTime() - new Date(a.fecha + ' ' + a.hora).getTime()
      )[0];

      const status = getAirQualityStatus(latest.pm2_5, latest.pm10, latest.no2);
      const aqi = this.calculateAQI(latest.pm2_5, latest.pm10, latest.no2);

      stations.push({
        name: stationName,
        distrito: latest.distrito,
        coordinates: getDistrictCoordinates(latest.distrito),
        currentAQI: aqi,
        status
      });
    });

    return stations;
  }

  async getTimeSeriesData(station: string, days: number = 7): Promise<TimeSeriesData[]> {
    const data = await this.loadAirQualityData();
    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - (days * 24 * 60 * 60 * 1000));

    return data
      .filter(record => {
        const recordDate = new Date(record.fecha);
        return record.estacion === station && 
               recordDate >= startDate && 
               recordDate <= endDate;
      })
      .map(record => ({
        timestamp: `${record.fecha} ${record.hora}`,
        pm10: record.pm10,
        pm2_5: record.pm2_5,
        no2: record.no2,
        station: record.estacion
      }))
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
  }

  async getForecastData(station: string): Promise<ForecastData[]> {
    // En producción, esto vendría de tu modelo de ML (Prophet/LSTM)
    // Por ahora simulamos datos de pronóstico
    const forecasts: ForecastData[] = [];
    const now = new Date();

    for (let i = 1; i <= 48; i++) {
      const futureTime = new Date(now.getTime() + (i * 60 * 60 * 1000));
      forecasts.push({
        timestamp: futureTime.toISOString(),
        predicted_pm10: 50 + Math.random() * 30 - 15, // Simular variación
        predicted_pm2_5: 25 + Math.random() * 20 - 10,
        predicted_no2: 30 + Math.random() * 15 - 7,
        station,
        confidence: 0.8 + Math.random() * 0.15 // 80-95% confidence
      });
    }

    return forecasts;
  }

  private calculateAQI(pm25: number | null, pm10: number | null, no2: number | null): number {
    // Simplified AQI calculation based on PM2.5
    const pm25Value = pm25 || 0;
    
    if (pm25Value <= 12) return Math.round(pm25Value * 50 / 12);
    if (pm25Value <= 35) return Math.round(50 + (pm25Value - 12) * 50 / 23);
    if (pm25Value <= 55) return Math.round(100 + (pm25Value - 35) * 50 / 20);
    if (pm25Value <= 150) return Math.round(150 + (pm25Value - 55) * 100 / 95);
    if (pm25Value <= 250) return Math.round(200 + (pm25Value - 150) * 100 / 100);
    return Math.round(300 + (pm25Value - 250) * 100 / 250);
  }

  private getMockData(): AirQualityData[] {
    // Datos simulados basados en la estructura real
    const stations = ['CAMPO_DE_MARTE', 'ATE', 'SAN_MARTIN_DE_PORRES'];
    const districts = ['JESUS_MARIA', 'ATE', 'SAN_MARTIN_DE_PORRES'];
    const data: AirQualityData[] = [];

    for (let day = 0; day < 7; day++) {
      for (let hour = 0; hour < 24; hour++) {
        stations.forEach((station, index) => {
          const date = new Date();
          date.setDate(date.getDate() - day);
          date.setHours(hour, 0, 0, 0);

          const baseValue = 30 + Math.sin(hour * Math.PI / 12) * 10; // Patrón diario
          
          data.push({
            id: data.length + 1,
            estacion: station,
            fecha: date.toISOString().split('T')[0].replace(/-/g, ''),
            hora: hour.toString().padStart(2, '0') + '0000',
            longitud: -77.0432 + (Math.random() - 0.5) * 0.1,
            latitud: -12.0705 + (Math.random() - 0.5) * 0.1,
            altitud: 117 + Math.random() * 100,
            pm10: baseValue * 2 + Math.random() * 20,
            pm2_5: baseValue + Math.random() * 15,
            no2: 25 + Math.random() * 20,
            departamento: 'LIMA',
            provincia: 'LIMA',
            distrito: districts[index],
            ubigeo: '150113',
            fecha_corte: '20240531'
          });
        });
      }
    }

    return data;
  }
}

export const dataService = new DataService();
