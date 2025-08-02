#!/usr/bin/env python3
"""
Pipeline simplificado de Fase 2 para datos disponibles
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_cleaned_data():
    """Procesa los datos ya limpios"""
    logger.info("🚀 PROCESANDO DATOS LIMPIOS - FASE 2 SIMPLIFICADA")
    logger.info("="*60)
    
    # Cargar datos limpios
    df = pd.read_csv("data/lima_air_quality_cleaned.csv")
    logger.info(f"📊 Datos cargados: {len(df):,} registros")
    
    # Convertir timestamp correctamente
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['date'] = df['timestamp'].dt.date
    
    # Crear características temporales básicas
    logger.info("🔧 Creando características temporales...")
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Estaciones del año (hemisferio sur)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'summer'
        elif month in [3, 4, 5]:
            return 'autumn'
        elif month in [6, 7, 8]:
            return 'winter'
        else:
            return 'spring'
    
    df['season'] = df['month'].apply(get_season)
    df['is_dry_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
    
    # Calcular AQI para PM10
    if 'pm10' in df.columns:
        df['pm10_aqi'] = df['pm10'].apply(calculate_pm10_aqi)
    
    # Crear ratios y índices
    if 'pm10' in df.columns and 'no2' in df.columns:
        df['pm10_no2_ratio'] = df['pm10'] / (df['no2'] + 1e-6)
    
    # Guardar dataset con características
    df.to_csv("data/lima_air_quality_features.csv", index=False)
    logger.info(f"✅ Dataset con características guardado: {len(df.columns)} columnas")
    
    # Crear agregaciones diarias
    logger.info("📈 Creando agregaciones diarias...")
    daily_df = create_daily_aggregation(df)
    daily_df.to_csv("data/aggregated/lima_air_quality_daily.csv", index=False)
    logger.info(f"✅ Agregación diaria: {len(daily_df):,} registros")
    
    # Crear agregaciones mensuales
    logger.info("📈 Creando agregaciones mensuales...")
    monthly_df = create_monthly_aggregation(df)
    monthly_df.to_csv("data/aggregated/lima_air_quality_monthly.csv", index=False)
    logger.info(f"✅ Agregación mensual: {len(monthly_df):,} registros")
    
    # Crear resumen por estación
    logger.info("🏭 Creando resumen por estación...")
    station_df = create_station_summary(df)
    station_df.to_csv("data/aggregated/lima_air_quality_station_summary.csv", index=False)
    logger.info(f"✅ Resumen de estaciones: {len(station_df):,} estaciones")
    
    # Generar reporte final
    generate_final_report(df, daily_df, monthly_df, station_df)
    
    return df

def calculate_pm10_aqi(pm10):
    """Calcula AQI para PM10"""
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

def create_daily_aggregation(df):
    """Crea agregación diaria"""
    os.makedirs("data/aggregated", exist_ok=True)
    
    # Contaminantes disponibles
    pollutants = ['pm10', 'no2']
    available = [col for col in pollutants if col in df.columns and df[col].notna().sum() > 0]
    
    print(f"Contaminantes disponibles: {available}")
    print(f"Columnas en df: {list(df.columns)}")
    print(f"Primeras fechas: {df['date'].head()}")
    
    if not available:
        print("No hay contaminantes disponibles")
        return pd.DataFrame()
    
    # Crear agregación básica primero
    try:
        daily_df = df.groupby(['station_name', 'date']).agg({
            'timestamp': 'first',
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()
        
        # Agregar estadísticas de contaminantes
        for col in available:
            daily_df[f'{col}_mean'] = df.groupby(['station_name', 'date'])[col].mean().values
            daily_df[f'{col}_max'] = df.groupby(['station_name', 'date'])[col].max().values
            daily_df[f'{col}_min'] = df.groupby(['station_name', 'date'])[col].min().values
            daily_df[f'{col}_count'] = df.groupby(['station_name', 'date'])[col].count().values
        
    except Exception as e:
        print(f"Error en agregación: {str(e)}")
        return pd.DataFrame()
    
    # Añadir características temporales
    daily_df['timestamp'] = pd.to_datetime(daily_df['timestamp'])
    daily_df['year'] = daily_df['timestamp'].dt.year
    daily_df['month'] = daily_df['timestamp'].dt.month
    daily_df['day_of_week'] = daily_df['timestamp'].dt.dayofweek
    daily_df['is_weekend'] = (daily_df['day_of_week'] >= 5).astype(int)
    
    return daily_df

def create_monthly_aggregation(df):
    """Crea agregación mensual"""
    # Contaminantes disponibles
    pollutants = ['pm10', 'no2']
    available = [col for col in pollutants if col in df.columns and df[col].notna().sum() > 0]
    
    if not available:
        return pd.DataFrame()
    
    # Crear año-mes
    df['year_month'] = df['timestamp'].dt.to_period('M').astype(str)
    
    # Agregación básica
    monthly_df = df.groupby(['station_name', 'year_month']).agg({
        'timestamp': 'first',
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()
    
    # Agregar estadísticas de contaminantes
    for col in available:
        monthly_df[f'{col}_mean'] = df.groupby(['station_name', 'year_month'])[col].mean().values
        monthly_df[f'{col}_max'] = df.groupby(['station_name', 'year_month'])[col].max().values
        monthly_df[f'{col}_min'] = df.groupby(['station_name', 'year_month'])[col].min().values
        monthly_df[f'{col}_median'] = df.groupby(['station_name', 'year_month'])[col].median().values
        monthly_df[f'{col}_std'] = df.groupby(['station_name', 'year_month'])[col].std().values
    
    # Añadir características
    monthly_df['timestamp'] = pd.to_datetime(monthly_df['timestamp'])
    monthly_df['year'] = monthly_df['timestamp'].dt.year
    monthly_df['month'] = monthly_df['timestamp'].dt.month
    
    return monthly_df

def create_station_summary(df):
    """Crea resumen por estación"""
    pollutants = ['pm10', 'no2']
    available = [col for col in pollutants if col in df.columns]
    
    agg_dict = {
        'latitude': 'first',
        'longitude': 'first',
        'timestamp': ['min', 'max', 'count']
    }
    
    for col in available:
        agg_dict[col] = ['mean', 'max', 'min', 'std', 'count']
    
    station_df = df.groupby('station_name').agg(agg_dict).reset_index()
    
    # Aplanar columnas
    station_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                         for col in station_df.columns.values]
    
    # Calcular período de operación
    if 'timestamp_min' in station_df.columns and 'timestamp_max' in station_df.columns:
        station_df['timestamp_min'] = pd.to_datetime(station_df['timestamp_min'])
        station_df['timestamp_max'] = pd.to_datetime(station_df['timestamp_max'])
        station_df['operation_days'] = (
            station_df['timestamp_max'] - station_df['timestamp_min']
        ).dt.days
    
    return station_df

def generate_final_report(df, daily_df, monthly_df, station_df):
    """Genera reporte final"""
    logger.info("\n" + "="*60)
    logger.info("🎯 REPORTE FINAL - FASE 2 COMPLETADA")
    logger.info("="*60)
    
    # Estadísticas generales
    logger.info(f"📊 DATOS PROCESADOS:")
    logger.info(f"  • Registros totales: {len(df):,}")
    logger.info(f"  • Estaciones: {df['station_name'].nunique()}")
    logger.info(f"  • Período: {df['timestamp'].min().strftime('%Y-%m-%d')} → {df['timestamp'].max().strftime('%Y-%m-%d')}")
    
    # Contaminantes
    pollutants = ['pm10', 'no2']
    available = [col for col in pollutants if col in df.columns]
    logger.info(f"\n💨 CONTAMINANTES PROCESADOS:")
    for col in available:
        non_null = df[col].notna().sum()
        mean_val = df[col].mean()
        logger.info(f"  • {col.upper()}: {non_null:,} mediciones (promedio: {mean_val:.1f} µg/m³)")
    
    # Agregaciones
    logger.info(f"\n📈 AGREGACIONES CREADAS:")
    logger.info(f"  • Datos con características: {len(df.columns)} columnas")
    logger.info(f"  • Agregación diaria: {len(daily_df):,} registros")
    logger.info(f"  • Agregación mensual: {len(monthly_df):,} registros")
    logger.info(f"  • Resumen de estaciones: {len(station_df):,} estaciones")
    
    # Archivos generados
    files = [
        "data/lima_air_quality_features.csv",
        "data/aggregated/lima_air_quality_daily.csv",
        "data/aggregated/lima_air_quality_monthly.csv",
        "data/aggregated/lima_air_quality_station_summary.csv"
    ]
    
    logger.info(f"\n📁 ARCHIVOS GENERADOS:")
    for file_path in files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"  ✅ {file_path}: {size_mb:.1f} MB")
        else:
            logger.info(f"  ❌ {file_path}: No encontrado")
    
    logger.info(f"\n🚀 FASE 2 COMPLETADA - Listo para Fase 3: Modelado ML")
    logger.info("="*60)

def main():
    """Función principal"""
    try:
        result_df = process_cleaned_data()
        logger.info("\n🎉 ¡PIPELINE FASE 2 COMPLETADO EXITOSAMENTE!")
        return True
    except Exception as e:
        logger.error(f"❌ Error en pipeline: {str(e)}")
        return False

if __name__ == "__main__":
    main()
