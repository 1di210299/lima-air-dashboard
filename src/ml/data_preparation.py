#!/usr/bin/env python3
"""
Preparación de datos para Machine Learning
Fase 3: Modelado y Predicción ML
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import logging

# Agregar el directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparator:
    """
    Clase para preparar datos para modelos de Machine Learning
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = {}
        self.target_columns = {}
        
    def prepare_regression_data(self, df: pd.DataFrame, target_column: str) -> Dict:
        """
        Prepara datos para modelos de regresión (predicción de contaminantes)
        
        Args:
            df: DataFrame con características
            target_column: Columna objetivo (pm10, no2, etc.)
            
        Returns:
            Diccionario con datos preparados
        """
        logger.info(f"Preparando datos para regresión - Target: {target_column}")
        
        # Filtrar datos donde existe el target
        df_clean = df[df[target_column].notna()].copy()
        logger.info(f"Datos con target válido: {len(df_clean):,} registros")
        
        # Seleccionar características
        feature_cols = self._select_regression_features(df_clean, target_column)
        logger.info(f"Características seleccionadas: {len(feature_cols)}")
        
        # Preparar características y target
        X = df_clean[feature_cols].copy()
        y = df_clean[target_column].copy()
        
        # Manejar valores faltantes
        X = self._handle_missing_values(X)
        
        # Codificar variables categóricas
        X = self._encode_categorical_features(X)
        
        # Dividir en train/test manteniendo orden temporal
        if 'timestamp' in df_clean.columns:
            df_clean = df_clean.sort_values('timestamp')
            split_idx = int(len(df_clean) * 0.8)
            
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            timestamps_train = df_clean['timestamp'].iloc[:split_idx]
            timestamps_test = df_clean['timestamp'].iloc[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            timestamps_train = timestamps_test = None
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[f'{target_column}_scaler'] = scaler
        self.feature_columns[target_column] = feature_cols
        
        logger.info(f"Train set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train.values,
            'y_test': y_test.values,
            'feature_names': feature_cols,
            'scaler': scaler,
            'timestamps_train': timestamps_train,
            'timestamps_test': timestamps_test,
            'train_indices': X_train.index,
            'test_indices': X_test.index
        }
    
    def prepare_classification_data(self, df: pd.DataFrame, target_column: str) -> Dict:
        """
        Prepara datos para clasificación de calidad del aire
        
        Args:
            df: DataFrame con características
            target_column: Columna de contaminante para crear clases de calidad
            
        Returns:
            Diccionario con datos preparados
        """
        logger.info(f"Preparando datos para clasificación - Basado en: {target_column}")
        
        # Filtrar datos válidos
        df_clean = df[df[target_column].notna()].copy()
        
        # Crear clases de calidad del aire
        df_clean['air_quality_class'] = self._create_air_quality_classes(
            df_clean[target_column], target_column
        )
        
        # Seleccionar características
        feature_cols = self._select_classification_features(df_clean, target_column)
        
        X = df_clean[feature_cols].copy()
        y = df_clean['air_quality_class'].copy()
        
        # Preparar datos
        X = self._handle_missing_values(X)
        X = self._encode_categorical_features(X)
        
        # Codificar clases
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        self.encoders[f'{target_column}_label_encoder'] = label_encoder
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Escalar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[f'{target_column}_classification_scaler'] = scaler
        self.feature_columns[f'{target_column}_classification'] = feature_cols
        
        logger.info(f"Clases de calidad: {list(label_encoder.classes_)}")
        logger.info(f"Distribución de clases: {np.bincount(y_encoded)}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_cols,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'class_names': list(label_encoder.classes_)
        }
    
    def prepare_timeseries_data(self, df: pd.DataFrame, target_column: str, 
                               sequence_length: int = 24) -> Dict:
        """
        Prepara datos para modelos de series temporales
        
        Args:
            df: DataFrame con datos temporales
            target_column: Columna objetivo
            sequence_length: Longitud de la secuencia (horas anteriores)
            
        Returns:
            Diccionario con secuencias preparadas
        """
        logger.info(f"Preparando datos de series temporales - Target: {target_column}")
        logger.info(f"Longitud de secuencia: {sequence_length} períodos")
        
        # Ordenar por timestamp y estación
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values(['station_name', 'timestamp']).copy()
        else:
            df_sorted = df.copy()
        
        # Filtrar datos válidos
        df_clean = df_sorted[df_sorted[target_column].notna()].copy()
        
        sequences = []
        targets = []
        station_sequences = []
        
        # Crear secuencias por estación
        for station in df_clean['station_name'].unique():
            station_data = df_clean[df_clean['station_name'] == station][target_column].values
            
            if len(station_data) < sequence_length + 1:
                continue
            
            for i in range(len(station_data) - sequence_length):
                sequences.append(station_data[i:i + sequence_length])
                targets.append(station_data[i + sequence_length])
                station_sequences.append(station)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        logger.info(f"Secuencias creadas: {len(sequences):,}")
        logger.info(f"Shape de secuencias: {sequences.shape}")
        
        # Dividir en train/test
        split_idx = int(len(sequences) * 0.8)
        
        X_train = sequences[:split_idx]
        X_test = sequences[split_idx:]
        y_train = targets[:split_idx]
        y_test = targets[split_idx:]
        
        # Normalizar
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
        
        target_scaler = MinMaxScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).ravel()
        
        self.scalers[f'{target_column}_timeseries_scaler'] = scaler
        self.scalers[f'{target_column}_target_scaler'] = target_scaler
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_scaled,
            'y_test': y_test_scaled,
            'sequence_length': sequence_length,
            'scaler': scaler,
            'target_scaler': target_scaler,
            'original_y_train': y_train,
            'original_y_test': y_test
        }
    
    def _select_regression_features(self, df: pd.DataFrame, target_column: str) -> List[str]:
        """Selecciona características para regresión"""
        
        # Características temporales
        temporal_features = [
            'hour', 'day_of_week', 'month', 'year',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'is_weekend', 'is_rush_hour', 'is_dry_season'
        ]
        
        # Características de lag y rolling
        lag_features = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
        
        # Otras características meteorológicas y de contaminantes
        pollution_features = [
            col for col in df.columns 
            if any(pollutant in col for pollutant in ['pm10', 'pm25', 'no2', 'so2', 'o3', 'co'])
            and col != target_column
            and 'aqi' not in col  # Excluir AQI para evitar data leakage
        ]
        
        # Características geográficas
        geo_features = ['latitude', 'longitude'] if 'latitude' in df.columns else []
        
        # Combinar todas las características disponibles
        all_features = temporal_features + lag_features + pollution_features + geo_features
        
        # Filtrar solo las que existen en el DataFrame
        available_features = [col for col in all_features if col in df.columns]
        
        # Remover características con demasiados valores faltantes
        valid_features = []
        for col in available_features:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct < 0.8:  # Menos del 80% de valores faltantes
                valid_features.append(col)
        
        return valid_features
    
    def _select_classification_features(self, df: pd.DataFrame, target_column: str) -> List[str]:
        """Selecciona características para clasificación"""
        # Similar a regresión pero incluye más características contextuales
        features = self._select_regression_features(df, target_column)
        
        # Añadir características adicionales para clasificación
        additional_features = [
            col for col in df.columns 
            if any(term in col for term in ['season', 'quality', 'index'])
            and col not in features
        ]
        
        return features + [col for col in additional_features if col in df.columns]
    
    def _create_air_quality_classes(self, values: pd.Series, pollutant: str) -> pd.Series:
        """Crea clases de calidad del aire basadas en valores del contaminante"""
        
        if pollutant == 'pm10':
            # Basado en estándares WHO y EPA para PM10
            conditions = [
                values <= 50,
                (values > 50) & (values <= 100),
                (values > 100) & (values <= 150),
                values > 150
            ]
            choices = ['Good', 'Moderate', 'Unhealthy', 'Hazardous']
            
        elif pollutant == 'no2':
            # Basado en estándares para NO2
            conditions = [
                values <= 25,
                (values > 25) & (values <= 50),
                (values > 50) & (values <= 100),
                values > 100
            ]
            choices = ['Good', 'Moderate', 'Unhealthy', 'Hazardous']
            
        else:
            # Clasificación genérica basada en percentiles
            q25, q50, q75 = values.quantile([0.25, 0.5, 0.75])
            conditions = [
                values <= q25,
                (values > q25) & (values <= q50),
                (values > q50) & (values <= q75),
                values > q75
            ]
            choices = ['Good', 'Moderate', 'Unhealthy', 'Hazardous']
        
        return pd.Series(np.select(conditions, choices, default='Unknown'), index=values.index)
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Maneja valores faltantes"""
        # Para características numéricas, usar mediana
        numeric_features = X.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 0:
            imputer = SimpleImputer(strategy='median')
            X[numeric_features] = imputer.fit_transform(X[numeric_features])
        
        # Para características categóricas, usar moda
        categorical_features = X.select_dtypes(include=['object']).columns
        if len(categorical_features) > 0:
            for col in categorical_features:
                mode_value = X[col].mode()
                if len(mode_value) > 0:
                    X[col].fillna(mode_value[0], inplace=True)
                else:
                    X[col].fillna('Unknown', inplace=True)
        
        return X
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Codifica características categóricas"""
        categorical_features = X.select_dtypes(include=['object']).columns
        
        for col in categorical_features:
            if col not in self.encoders:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col].astype(str))
                self.encoders[col] = encoder
            else:
                # Usar encoder existente
                encoder = self.encoders[col]
                # Manejar valores nuevos
                unique_values = set(X[col].astype(str))
                known_values = set(encoder.classes_)
                new_values = unique_values - known_values
                
                if new_values:
                    # Asignar valores nuevos a la clase más común
                    most_common_class = encoder.classes_[0]
                    X[col] = X[col].astype(str).replace(list(new_values), most_common_class)
                
                X[col] = encoder.transform(X[col].astype(str))
        
        return X

def main():
    """Función principal para pruebas"""
    try:
        # Cargar datos con características
        df = pd.read_csv("data/lima_air_quality_features.csv")
        logger.info(f"Datos cargados: {len(df):,} registros")
        
        preparator = DataPreparator()
        
        # Preparar datos para regresión de PM10
        if 'pm10' in df.columns:
            pm10_data = preparator.prepare_regression_data(df, 'pm10')
            logger.info("✅ Datos de regresión PM10 preparados")
        
        # Preparar datos para clasificación
        if 'pm10' in df.columns:
            classification_data = preparator.prepare_classification_data(df, 'pm10')
            logger.info("✅ Datos de clasificación preparados")
        
        # Preparar datos de series temporales
        if 'pm10' in df.columns and 'timestamp' in df.columns:
            timeseries_data = preparator.prepare_timeseries_data(df, 'pm10')
            logger.info("✅ Datos de series temporales preparados")
        
        logger.info("🎉 Preparación de datos completada")
        
    except FileNotFoundError:
        logger.error("Archivo con características no encontrado. Ejecutar primero Fase 2")
    except Exception as e:
        logger.error(f"Error en preparación de datos: {str(e)}")

if __name__ == "__main__":
    main()
