#!/usr/bin/env python3
"""
Sistema de agregaciones temporales
Fase 2: Preprocesamiento y limpieza de datos
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

# Agregar el directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesAggregator:
    """
    Clase para crear agregaciones temporales de datos de calidad del aire
    """
    
    def __init__(self):
        self.aggregated_datasets = {}
    
    def create_all_aggregations(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Crea todas las agregaciones temporales
        
        Args:
            df: DataFrame con datos por hora
            
        Returns:
            Diccionario con diferentes agregaciones
        """
        logger.info("Iniciando creación de agregaciones temporales")
        
        # Asegurar que timestamp es datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # Remover filas con timestamp inválido
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values(['station_name', 'timestamp'])
        else:
            logger.warning("No hay columna timestamp disponible")
            return {}
        
        # Crear diferentes agregaciones
        self.aggregated_datasets['daily'] = self._create_daily_aggregation(df)
        self.aggregated_datasets['weekly'] = self._create_weekly_aggregation(df)
        self.aggregated_datasets['monthly'] = self._create_monthly_aggregation(df)
        self.aggregated_datasets['station_summary'] = self._create_station_summary(df)
        
        logger.info(f"✅ Creadas {len(self.aggregated_datasets)} agregaciones temporales")
        
        return self.aggregated_datasets
    
    def _create_daily_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea agregación diaria"""
        logger.info("Creando agregación diaria")
        
        if 'timestamp' not in df.columns:
            logger.warning("No hay columna timestamp para agregación diaria")
            return pd.DataFrame()
        
        # Columnas de contaminantes
        pollutant_cols = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
        available_pollutants = [col for col in pollutant_cols if col in df.columns]
        
        # Agregar fecha
        df['date'] = df['timestamp'].dt.date
        
        # Realizar agregación básica
        daily_df = df.groupby(['station_name', 'date']).agg({
            'timestamp': 'min',
            'latitude': 'first',
            'longitude': 'first',
            **{col: ['mean', 'max', 'min', 'std', 'count'] for col in available_pollutants}
        }).reset_index()
        
        # Aplanar columnas multi-nivel
        daily_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                           for col in daily_df.columns.values]
        
        # Añadir características temporales diarias
        daily_df['timestamp'] = pd.to_datetime(daily_df['timestamp'])
        daily_df['day_of_week'] = daily_df['timestamp'].dt.dayofweek
        daily_df['month'] = daily_df['timestamp'].dt.month
        daily_df['year'] = daily_df['timestamp'].dt.year
        daily_df['is_weekend'] = (daily_df['day_of_week'] >= 5).astype(int)
        
        # Calcular índice de calidad diario
        if 'pm25_mean' in daily_df.columns:
            daily_df['daily_aqi_pm25'] = self._calculate_daily_aqi(daily_df['pm25_mean'])
        elif 'pm10_mean' in daily_df.columns:
            daily_df['daily_aqi_pm10'] = self._calculate_daily_aqi(daily_df['pm10_mean'])
        
        logger.info(f"Agregación diaria creada: {len(daily_df):,} registros")
        return daily_df
    
    def _create_weekly_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea agregación semanal"""
        logger.info("Creando agregación semanal")
        
        if 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        pollutant_cols = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
        available_pollutants = [col for col in pollutant_cols if col in df.columns]
        
        # Agregar semana
        df['year_week'] = df['timestamp'].dt.strftime('%Y-W%U')
        
        # Realizar agregación
        weekly_df = df.groupby(['station_name', 'year_week']).agg({
            'timestamp': 'min',
            'latitude': 'first',
            'longitude': 'first',
            **{col: ['mean', 'max', 'min'] for col in available_pollutants}
        }).reset_index()
        
        # Aplanar columnas
        weekly_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                            for col in weekly_df.columns.values]
        
        weekly_df['timestamp'] = pd.to_datetime(weekly_df['timestamp'])
        weekly_df['month'] = weekly_df['timestamp'].dt.month
        weekly_df['year'] = weekly_df['timestamp'].dt.year
        
        logger.info(f"Agregación semanal creada: {len(weekly_df):,} registros")
        return weekly_df
    
    def _create_monthly_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea agregación mensual"""
        logger.info("Creando agregación mensual")
        
        if 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        pollutant_cols = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
        available_pollutants = [col for col in pollutant_cols if col in df.columns]
        
        # Agregar año-mes
        df['year_month'] = df['timestamp'].dt.to_period('M')
        
        # Realizar agregación
        monthly_df = df.groupby(['station_name', 'year_month']).agg({
            'timestamp': 'min',
            'latitude': 'first',
            'longitude': 'first',
            **{col: ['mean', 'max', 'min', 'std', 'count'] for col in available_pollutants}
        }).reset_index()
        
        # Aplanar columnas
        monthly_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                             for col in monthly_df.columns.values]
        
        monthly_df['timestamp'] = pd.to_datetime(monthly_df['timestamp'])
        monthly_df['month'] = monthly_df['timestamp'].dt.month
        monthly_df['year'] = monthly_df['timestamp'].dt.year
        
        # Añadir características estacionales
        monthly_df['season'] = monthly_df['month'].apply(self._get_season)
        monthly_df['is_dry_season'] = monthly_df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
        
        logger.info(f"Agregación mensual creada: {len(monthly_df):,} registros")
        return monthly_df
    
    def _create_station_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea resumen por estación"""
        logger.info("Creando resumen por estación")
        
        pollutant_cols = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
        available_pollutants = [col for col in pollutant_cols if col in df.columns]
        
        agg_functions = {
            'latitude': 'first',
            'longitude': 'first',
            'timestamp': ['min', 'max', 'count']
        }
        
        for col in available_pollutants:
            agg_functions[col] = ['mean', 'max', 'min', 'std', 'count']
        
        station_df = df.groupby('station_name').agg(agg_functions).reset_index()
        
        # Aplanar columnas multi-nivel
        station_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                             for col in station_df.columns.values]
        
        # Renombrar algunas columnas importantes
        if 'timestamp_min' in station_df.columns:
            station_df = station_df.rename(columns={
                'timestamp_min': 'first_measurement',
                'timestamp_max': 'last_measurement',
                'timestamp_count': 'total_measurements'
            })
        
        # Calcular período de operación
        if 'first_measurement' in station_df.columns and 'last_measurement' in station_df.columns:
            station_df['first_measurement'] = pd.to_datetime(station_df['first_measurement'])
            station_df['last_measurement'] = pd.to_datetime(station_df['last_measurement'])
            station_df['operation_days'] = (
                station_df['last_measurement'] - station_df['first_measurement']
            ).dt.days
        
        # Calcular completitud de datos
        if 'total_measurements' in station_df.columns and 'operation_days' in station_df.columns:
            expected_measurements = station_df['operation_days'] * 24  # 24 mediciones por día
            station_df['data_completeness'] = (
                station_df['total_measurements'] / expected_measurements
            ).round(3)
        
        logger.info(f"Resumen por estación creado: {len(station_df):,} estaciones")
        return station_df
    
    def _calculate_daily_aqi(self, pm25_daily: pd.Series) -> pd.Series:
        """Calcula AQI diario para PM2.5"""
        def pm25_to_aqi(pm25):
            if pd.isna(pm25):
                return np.nan
            elif pm25 <= 12:
                return pm25 * 50 / 12
            elif pm25 <= 35.4:
                return 50 + (pm25 - 12) * 50 / (35.4 - 12)
            elif pm25 <= 55.4:
                return 100 + (pm25 - 35.4) * 50 / (55.4 - 35.4)
            elif pm25 <= 150.4:
                return 150 + (pm25 - 55.4) * 50 / (150.4 - 55.4)
            else:
                return 200 + (pm25 - 150.4) * 100 / (250.4 - 150.4)
        
        return pm25_daily.apply(pm25_to_aqi)
    
    def _get_season(self, month: int) -> str:
        """Determina la estación del año para el hemisferio sur"""
        if month in [12, 1, 2]:
            return 'summer'
        elif month in [3, 4, 5]:
            return 'autumn'
        elif month in [6, 7, 8]:
            return 'winter'
        else:
            return 'spring'
    
    def save_aggregations(self, output_dir: str = "data/aggregated"):
        """Guarda todas las agregaciones en archivos separados"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for name, df in self.aggregated_datasets.items():
            if not df.empty:
                output_file = os.path.join(output_dir, f"lima_air_quality_{name}.csv")
                df.to_csv(output_file, index=False)
                logger.info(f"Guardado: {output_file} ({len(df):,} registros)")
    
    def get_aggregation_summary(self) -> Dict:
        """Retorna resumen de las agregaciones creadas"""
        summary = {}
        for name, df in self.aggregated_datasets.items():
            if not df.empty:
                summary[name] = {
                    'records': len(df),
                    'columns': len(df.columns),
                    'date_range': (
                        df['timestamp'].min().strftime('%Y-%m-%d') if 'timestamp' in df.columns else 'N/A',
                        df['timestamp'].max().strftime('%Y-%m-%d') if 'timestamp' in df.columns else 'N/A'
                    ),
                    'stations': df['station_name'].nunique() if 'station_name' in df.columns else 0
                }
        return summary

def main():
    """Función principal para pruebas"""
    try:
        # Cargar datos con características
        df = pd.read_csv("data/lima_air_quality_features.csv")
        logger.info(f"Cargados {len(df):,} registros con características")
        
        # Crear agregaciones
        aggregator = TimeSeriesAggregator()
        aggregations = aggregator.create_all_aggregations(df)
        
        # Guardar agregaciones
        aggregator.save_aggregations()
        
        # Mostrar resumen
        summary = aggregator.get_aggregation_summary()
        
        logger.info("\n" + "="*60)
        logger.info("RESUMEN DE AGREGACIONES TEMPORALES")
        logger.info("="*60)
        
        for name, info in summary.items():
            logger.info(f"\n📊 {name.upper()}:")
            logger.info(f"  • Registros: {info['records']:,}")
            logger.info(f"  • Columnas: {info['columns']}")
            logger.info(f"  • Período: {info['date_range'][0]} → {info['date_range'][1]}")
            logger.info(f"  • Estaciones: {info['stations']}")
        
        logger.info("="*60)
        
    except FileNotFoundError:
        logger.error("Archivo con características no encontrado. Ejecutar primero feature_engineer.py")

if __name__ == "__main__":
    main()
