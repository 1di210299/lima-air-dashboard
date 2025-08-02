#!/usr/bin/env python3
"""
Sistema de ingeniería de características (Feature Engineering)
Fase 2: Preprocesamiento y limpieza de datos
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

# Agregar el directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Clase para crear características derivadas de los datos de calidad del aire
    """
    
    def __init__(self):
        self.features_created = []
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea todas las características derivadas
        
        Args:
            df: DataFrame con datos limpios
            
        Returns:
            DataFrame con características adicionales
        """
        logger.info("Iniciando creación de características derivadas")
        
        df = self._create_temporal_features(df)
        df = self._create_air_quality_indices(df)
        df = self._create_pollution_ratios(df)
        df = self._create_seasonal_features(df)
        df = self._create_lag_features(df)
        df = self._create_statistical_features(df)
        
        logger.info(f"✅ Creadas {len(self.features_created)} características nuevas")
        logger.info(f"Características: {', '.join(self.features_created)}")
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características temporales"""
        logger.info("Creando características temporales")
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Características básicas de tiempo
            df['year'] = df['timestamp'].dt.year
            df['month'] = df['timestamp'].dt.month
            df['day'] = df['timestamp'].dt.day
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_year'] = df['timestamp'].dt.dayofyear
            df['week_of_year'] = df['timestamp'].dt.isocalendar().week
            
            # Características cíclicas (sin y cos para capturar periodicidad)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Categorías temporales
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
            df['is_business_hour'] = df['hour'].between(9, 17).astype(int)
            
            self.features_created.extend([
                'year', 'month', 'day', 'hour', 'day_of_week', 'day_of_year', 'week_of_year',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
                'is_weekend', 'is_rush_hour', 'is_night', 'is_business_hour'
            ])
        
        return df
    
    def _create_air_quality_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea índices de calidad del aire"""
        logger.info("Creando índices de calidad del aire")
        
        # Índice de Calidad del Aire (AQI) simplificado
        if 'pm25' in df.columns:
            df['pm25_aqi'] = self._calculate_aqi_pm25(df['pm25'])
            self.features_created.append('pm25_aqi')
        
        if 'pm10' in df.columns:
            df['pm10_aqi'] = self._calculate_aqi_pm10(df['pm10'])
            self.features_created.append('pm10_aqi')
        
        # Índice compuesto de contaminación
        pollutant_cols = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
        available_pollutants = [col for col in pollutant_cols if col in df.columns]
        
        if len(available_pollutants) >= 2:
            # Normalizar contaminantes (0-1) y promediar
            df_norm = df[available_pollutants].copy()
            for col in available_pollutants:
                if df_norm[col].notna().sum() > 0:
                    df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
            
            df['pollution_index'] = df_norm.mean(axis=1, skipna=True)
            self.features_created.append('pollution_index')
        
        return df
    
    def _create_pollution_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea ratios entre contaminantes"""
        logger.info("Creando ratios de contaminantes")
        
        # Ratio PM2.5/PM10 (indicador de fuentes de contaminación)
        if 'pm25' in df.columns and 'pm10' in df.columns:
            df['pm25_pm10_ratio'] = df['pm25'] / (df['pm10'] + 1e-6)  # Evitar división por cero
            self.features_created.append('pm25_pm10_ratio')
        
        # Ratio NO2/O3 (indicador de balance oxidante)
        if 'no2' in df.columns and 'o3' in df.columns:
            df['no2_o3_ratio'] = df['no2'] / (df['o3'] + 1e-6)
            self.features_created.append('no2_o3_ratio')
        
        # Suma de contaminantes primarios
        primary_pollutants = ['pm25', 'pm10', 'so2', 'no2', 'co']
        available_primary = [col for col in primary_pollutants if col in df.columns]
        if len(available_primary) >= 2:
            df['primary_pollutants_sum'] = df[available_primary].sum(axis=1, skipna=True)
            self.features_created.append('primary_pollutants_sum')
        
        return df
    
    def _create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características estacionales"""
        logger.info("Creando características estacionales")
        
        if 'month' in df.columns:
            # Estaciones del año en el hemisferio sur (Perú)
            def get_season(month):
                if month in [12, 1, 2]:
                    return 'summer'  # Verano
                elif month in [3, 4, 5]:
                    return 'autumn'  # Otoño
                elif month in [6, 7, 8]:
                    return 'winter'  # Invierno
                else:
                    return 'spring'  # Primavera
            
            df['season'] = df['month'].apply(get_season)
            
            # One-hot encoding para estaciones
            season_dummies = pd.get_dummies(df['season'], prefix='season')
            df = pd.concat([df, season_dummies], axis=1)
            
            self.features_created.extend(['season'] + list(season_dummies.columns))
        
        # Período seco/húmedo en Lima
        if 'month' in df.columns:
            df['is_dry_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)  # Mayo-Octubre
            df['is_wet_season'] = df['month'].isin([11, 12, 1, 2, 3, 4]).astype(int)  # Noviembre-Abril
            
            self.features_created.extend(['is_dry_season', 'is_wet_season'])
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características de lag temporal"""
        logger.info("Creando características de lag temporal")
        
        if 'station_name' in df.columns and 'timestamp' in df.columns:
            # Ordenar por estación y tiempo
            df = df.sort_values(['station_name', 'timestamp'])
            
            # Crear lags para contaminantes principales
            pollutant_cols = ['pm25', 'pm10', 'no2', 'o3']
            lag_periods = [1, 3, 6, 24]  # 1h, 3h, 6h, 24h
            
            for col in pollutant_cols:
                if col in df.columns:
                    for lag in lag_periods:
                        lag_col = f'{col}_lag_{lag}h'
                        df[lag_col] = df.groupby('station_name')[col].shift(lag)
                        self.features_created.append(lag_col)
            
            # Rolling means (promedios móviles)
            for col in pollutant_cols:
                if col in df.columns:
                    for window in [3, 6, 12, 24]:
                        rolling_col = f'{col}_rolling_{window}h'
                        df[rolling_col] = df.groupby('station_name')[col].rolling(
                            window=window, min_periods=1
                        ).mean().reset_index(0, drop=True)
                        self.features_created.append(rolling_col)
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características estadísticas"""
        logger.info("Creando características estadísticas")
        
        if 'station_name' in df.columns:
            pollutant_cols = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
            available_pollutants = [col for col in pollutant_cols if col in df.columns]
            
            for col in available_pollutants:
                # Rolling statistics (ventana de 24 horas)
                rolling_std = f'{col}_rolling_std_24h'
                rolling_max = f'{col}_rolling_max_24h'
                rolling_min = f'{col}_rolling_min_24h'
                
                df[rolling_std] = df.groupby('station_name')[col].rolling(
                    window=24, min_periods=1
                ).std().reset_index(0, drop=True)
                
                df[rolling_max] = df.groupby('station_name')[col].rolling(
                    window=24, min_periods=1
                ).max().reset_index(0, drop=True)
                
                df[rolling_min] = df.groupby('station_name')[col].rolling(
                    window=24, min_periods=1
                ).min().reset_index(0, drop=True)
                
                self.features_created.extend([rolling_std, rolling_max, rolling_min])
                
                # Diferencia con respecto al promedio de la estación
                station_mean = f'{col}_station_mean_diff'
                df[station_mean] = df.groupby('station_name')[col].transform(lambda x: x - x.mean())
                self.features_created.append(station_mean)
        
        return df
    
    def _calculate_aqi_pm25(self, pm25_values: pd.Series) -> pd.Series:
        """Calcula AQI para PM2.5 (escala EPA)"""
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
            elif pm25 <= 250.4:
                return 200 + (pm25 - 150.4) * 100 / (250.4 - 150.4)
            else:
                return 300 + (pm25 - 250.4) * 100 / (350.4 - 250.4)
        
        return pm25_values.apply(pm25_to_aqi)
    
    def _calculate_aqi_pm10(self, pm10_values: pd.Series) -> pd.Series:
        """Calcula AQI para PM10 (escala EPA)"""
        def pm10_to_aqi(pm10):
            if pd.isna(pm10):
                return np.nan
            elif pm10 <= 54:
                return pm10 * 50 / 54
            elif pm10 <= 154:
                return 50 + (pm10 - 54) * 50 / (154 - 54)
            elif pm10 <= 254:
                return 100 + (pm10 - 154) * 50 / (254 - 154)
            elif pm10 <= 354:
                return 150 + (pm10 - 254) * 50 / (354 - 254)
            elif pm10 <= 424:
                return 200 + (pm10 - 354) * 100 / (424 - 354)
            else:
                return 300 + (pm10 - 424) * 100 / (504 - 424)
        
        return pm10_values.apply(pm10_to_aqi)

def main():
    """Función principal para pruebas"""
    # Cargar datos limpios
    try:
        df = pd.read_csv("data/lima_air_quality_cleaned.csv")
        logger.info(f"Cargados {len(df):,} registros limpios")
        
        # Crear características
        engineer = FeatureEngineer()
        df_with_features = engineer.create_all_features(df)
        
        # Guardar dataset con características
        output_file = "data/lima_air_quality_features.csv"
        df_with_features.to_csv(output_file, index=False)
        
        logger.info(f"✅ Dataset con características guardado: {output_file}")
        logger.info(f"📊 Columnas totales: {len(df_with_features.columns)}")
        
    except FileNotFoundError:
        logger.error("Archivo de datos limpios no encontrado. Ejecutar primero data_cleaner.py")

if __name__ == "__main__":
    main()
