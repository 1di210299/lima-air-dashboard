#!/usr/bin/env python3
"""
Sistema de limpieza y validación de datos
Fase 2: Preprocesamiento y limpieza de datos
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

# Agregar el directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.connection import DatabaseManager
from src.database.models import AirQualityMeasurement
from config.settings import POLLUTANT_LIMITS, VALIDATION_RULES

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Clase para limpiar y validar datos de calidad del aire
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.stats = {
            'total_records': 0,
            'cleaned_records': 0,
            'removed_duplicates': 0,
            'removed_outliers': 0,
            'imputed_values': 0,
            'invalid_records': 0
        }
    
    def clean_dataset(self, input_file: str = None, output_file: str = None) -> pd.DataFrame:
        """
        Limpia el dataset completo
        
        Args:
            input_file: Archivo de entrada (opcional, usa el principal por defecto)
            output_file: Archivo de salida limpio
            
        Returns:
            DataFrame limpio
        """
        logger.info("Iniciando limpieza completa del dataset")
        
        # Cargar datos
        if input_file is None:
            input_file = "data/lima_air_quality_complete.csv"
        
        logger.info(f"Cargando datos desde: {input_file}")
        df = pd.read_csv(input_file)
        self.stats['total_records'] = len(df)
        
        logger.info(f"Dataset cargado: {len(df):,} registros")
        
        # Pipeline de limpieza
        df = self._standardize_columns(df)
        df = self._clean_temporal_data(df)
        df = self._clean_geographical_data(df)
        df = self._remove_duplicates(df)
        df = self._validate_pollutant_values(df)
        df = self._remove_outliers(df)
        df = self._handle_missing_values(df)
        df = self._add_data_quality_flags(df)
        
        self.stats['cleaned_records'] = len(df)
        
        # Guardar dataset limpio
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Dataset limpio guardado en: {output_file}")
        
        self._print_cleaning_summary()
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estandariza nombres de columnas y tipos de datos"""
        logger.info("Estandarizando columnas y tipos de datos")
        
        # Mapear nombres de columnas inconsistentes
        column_mapping = {
            'ESTACION': 'station_name',
            'FECHA': 'date',
            'HORA': 'hour',
            'LONGITUD': 'longitude',
            'LATITUD': 'latitude',
            'PM25': 'pm25',
            'PM10': 'pm10',
            'SO2': 'so2',
            'NO2': 'no2',
            'O3': 'o3',
            'CO': 'co'
        }
        
        # Renombrar columnas si existen
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Convertir tipos de datos
        if 'longitude' in df.columns:
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        if 'latitude' in df.columns:
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        
        # Convertir contaminantes a numérico
        pollutant_columns = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
        for col in pollutant_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _clean_temporal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y valida datos temporales"""
        logger.info("Limpiando datos temporales")
        
        # Crear timestamp si no existe
        if 'timestamp' not in df.columns:
            if 'date' in df.columns and 'hour' in df.columns:
                # Combinar fecha y hora
                df['date_str'] = df['date'].astype(str)
                df['hour_str'] = df['hour'].astype(str).str.zfill(2) + ':00:00'
                df['timestamp'] = pd.to_datetime(
                    df['date_str'] + ' ' + df['hour_str'], 
                    format='%Y%m%d %H:%M:%S',
                    errors='coerce'
                )
                df = df.drop(['date_str', 'hour_str'], axis=1)
            elif 'FECHA' in df.columns and 'HORA' in df.columns:
                # Formato alternativo
                df['date_str'] = df['FECHA'].astype(str)
                df['hour_str'] = df['HORA'].astype(str).str.zfill(2) + ':00:00'
                df['timestamp'] = pd.to_datetime(
                    df['date_str'] + ' ' + df['hour_str'], 
                    format='%Y%m%d %H:%M:%S',
                    errors='coerce'
                )
                df = df.drop(['date_str', 'hour_str'], axis=1)
        else:
            # Convertir timestamp existente
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Validar timestamps
        if 'timestamp' in df.columns:
            # Remover timestamps inválidos
            before_count = len(df)
            df = df.dropna(subset=['timestamp'])
            removed = before_count - len(df)
            if removed > 0:
                logger.warning(f"Removidos {removed} registros con timestamps inválidos")
            
            # Validar rango temporal (últimos 20 años)
            min_date = datetime.now() - timedelta(days=20*365)
            max_date = datetime.now() + timedelta(days=1)
            
            mask = (df['timestamp'] >= min_date) & (df['timestamp'] <= max_date)
            before_count = len(df)
            df = df[mask]
            removed = before_count - len(df)
            if removed > 0:
                logger.warning(f"Removidos {removed} registros fuera del rango temporal válido")
        
        return df
    
    def _clean_geographical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y valida coordenadas geográficas"""
        logger.info("Validando coordenadas geográficas")
        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Límites aproximados de Lima Metropolitana
            LIMA_BOUNDS = {
                'lat_min': -12.5,
                'lat_max': -11.5,
                'lon_min': -77.5,
                'lon_max': -76.5
            }
            
            # Validar coordenadas dentro de Lima
            valid_coords = (
                (df['latitude'] >= LIMA_BOUNDS['lat_min']) &
                (df['latitude'] <= LIMA_BOUNDS['lat_max']) &
                (df['longitude'] >= LIMA_BOUNDS['lon_min']) &
                (df['longitude'] <= LIMA_BOUNDS['lon_max'])
            )
            
            before_count = len(df)
            df = df[valid_coords | df['latitude'].isna() | df['longitude'].isna()]
            removed = before_count - len(df)
            if removed > 0:
                logger.warning(f"Removidos {removed} registros con coordenadas fuera de Lima")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remueve registros duplicados"""
        logger.info("Removiendo registros duplicados")
        
        before_count = len(df)
        
        # Identificar columnas clave para duplicados
        key_columns = ['station_name', 'timestamp']
        if 'timestamp' not in df.columns:
            key_columns = ['station_name', 'date', 'hour']
        
        # Remover duplicados exactos
        df = df.drop_duplicates()
        
        # Remover duplicados por estación y tiempo (conservar el más completo)
        existing_key_columns = [col for col in key_columns if col in df.columns]
        if existing_key_columns:
            # Ordenar por número de valores no nulos (más completo primero)
            df['completeness'] = df.count(axis=1)
            df = df.sort_values('completeness', ascending=False)
            df = df.drop_duplicates(subset=existing_key_columns, keep='first')
            df = df.drop('completeness', axis=1)
        
        removed = before_count - len(df)
        self.stats['removed_duplicates'] = removed
        
        if removed > 0:
            logger.info(f"Removidos {removed:,} registros duplicados")
        
        return df
    
    def _validate_pollutant_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida y limpia valores de contaminantes"""
        logger.info("Validando valores de contaminantes")
        
        # Límites físicos razonables para contaminantes (µg/m³)
        limits = {
            'pm25': {'min': 0, 'max': 1000},
            'pm10': {'min': 0, 'max': 2000},
            'so2': {'min': 0, 'max': 1000},
            'no2': {'min': 0, 'max': 500},
            'o3': {'min': 0, 'max': 400},
            'co': {'min': 0, 'max': 50000}  # CO en µg/m³
        }
        
        invalid_count = 0
        
        for pollutant, limit in limits.items():
            if pollutant in df.columns:
                # Contar valores fuera del rango
                invalid_mask = (
                    (df[pollutant] < limit['min']) | 
                    (df[pollutant] > limit['max'])
                ) & df[pollutant].notna()
                
                invalid_values = invalid_mask.sum()
                if invalid_values > 0:
                    logger.warning(f"Encontrados {invalid_values} valores inválidos para {pollutant}")
                    # Convertir valores inválidos a NaN
                    df.loc[invalid_mask, pollutant] = np.nan
                    invalid_count += invalid_values
        
        self.stats['invalid_records'] = invalid_count
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remueve outliers usando método IQR"""
        logger.info("Removiendo outliers")
        
        pollutant_columns = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
        outlier_count = 0
        
        for col in pollutant_columns:
            if col in df.columns and df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Límites para outliers (más conservador: 3*IQR)
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Identificar outliers
                outlier_mask = (
                    (df[col] < lower_bound) | 
                    (df[col] > upper_bound)
                ) & df[col].notna()
                
                outliers = outlier_mask.sum()
                if outliers > 0:
                    logger.info(f"Removiendo {outliers} outliers de {col}")
                    df.loc[outlier_mask, col] = np.nan
                    outlier_count += outliers
        
        self.stats['removed_outliers'] = outlier_count
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maneja valores faltantes"""
        logger.info("Manejando valores faltantes")
        
        pollutant_columns = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
        imputed_count = 0
        
        # Para cada contaminante, imputar usando interpolación temporal por estación
        for col in pollutant_columns:
            if col in df.columns:
                before_na = df[col].isna().sum()
                
                if 'station_name' in df.columns and 'timestamp' in df.columns:
                    # Ordenar por estación y tiempo
                    df = df.sort_values(['station_name', 'timestamp'])
                    
                    # Interpolación lineal por estación
                    df[col] = df.groupby('station_name')[col].transform(
                        lambda x: x.interpolate(method='linear', limit=3)
                    )
                
                after_na = df[col].isna().sum()
                imputed = before_na - after_na
                imputed_count += imputed
                
                if imputed > 0:
                    logger.info(f"Imputados {imputed} valores faltantes para {col}")
        
        self.stats['imputed_values'] = imputed_count
        
        return df
    
    def _add_data_quality_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade flags de calidad de datos"""
        logger.info("Añadiendo flags de calidad de datos")
        
        # Flag de completitud de datos
        pollutant_columns = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
        existing_pollutants = [col for col in pollutant_columns if col in df.columns]
        
        if existing_pollutants:
            df['data_completeness'] = df[existing_pollutants].notna().sum(axis=1) / len(existing_pollutants)
            
            # Clasificación de calidad
            df['data_quality'] = pd.cut(
                df['data_completeness'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['low', 'medium', 'high'],
                include_lowest=True
            )
        
        return df
    
    def _print_cleaning_summary(self):
        """Imprime resumen de la limpieza"""
        logger.info("\n" + "="*60)
        logger.info("RESUMEN DE LIMPIEZA DE DATOS")
        logger.info("="*60)
        logger.info(f"📊 Registros originales: {self.stats['total_records']:,}")
        logger.info(f"✅ Registros finales: {self.stats['cleaned_records']:,}")
        logger.info(f"🗑️  Duplicados removidos: {self.stats['removed_duplicates']:,}")
        logger.info(f"📉 Outliers removidos: {self.stats['removed_outliers']:,}")
        logger.info(f"❌ Valores inválidos: {self.stats['invalid_records']:,}")
        logger.info(f"🔧 Valores imputados: {self.stats['imputed_values']:,}")
        
        retention_rate = (self.stats['cleaned_records'] / self.stats['total_records']) * 100
        logger.info(f"📈 Tasa de retención: {retention_rate:.1f}%")
        logger.info("="*60)

def main():
    """Función principal para ejecutar la limpieza"""
    cleaner = DataCleaner()
    
    # Limpiar dataset principal
    cleaned_df = cleaner.clean_dataset(
        input_file="data/lima_air_quality_complete.csv",
        output_file="data/lima_air_quality_cleaned.csv"
    )
    
    logger.info(f"✅ Limpieza completada. Dataset limpio: {len(cleaned_df):,} registros")

if __name__ == "__main__":
    main()
