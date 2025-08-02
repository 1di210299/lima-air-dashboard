"""
Proceso ETL (Extract, Transform, Load) para datos de calidad del aire
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from src.database.connection import get_db_manager
from config.settings import VALIDATION_CONFIG

logger = logging.getLogger(__name__)

class AirQualityETL:
    """
    Procesador ETL para datos de calidad del aire
    """
    
    def __init__(self, use_sqlite: bool = True):
        """
        Inicializa el procesador ETL
        
        Args:
            use_sqlite: Si usar SQLite en lugar de PostgreSQL
        """
        self.db_manager = get_db_manager(use_sqlite=use_sqlite)
        
    def extract_historical_data(self, file_path: str, chunk_size: int = 50000) -> pd.DataFrame:
        """
        Extrae datos del archivo histórico completo
        
        Args:
            file_path: Ruta del archivo de datos históricos
            chunk_size: Tamaño de chunks para procesar archivo grande
            
        Returns:
            DataFrame con datos extraídos
        """
        logger.info(f"Extrayendo datos de: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"Archivo no encontrado: {file_path}")
            return pd.DataFrame()
        
        try:
            # Leer archivo por chunks para manejar archivos grandes
            chunks = []
            chunk_count = 0
            
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunk_count += 1
                logger.info(f"Procesando chunk {chunk_count}: {len(chunk)} registros")
                
                # Transformar chunk
                transformed_chunk = self.transform_data(chunk)
                
                if len(transformed_chunk) > 0:
                    chunks.append(transformed_chunk)
                
                # Limitar número de chunks para demo (remover en producción)
                if chunk_count >= 5:  # Solo primeros 5 chunks para demo
                    logger.info("Limitando procesamiento a 5 chunks para demo")
                    break
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Extracción completada: {len(df)} registros totales")
                return df
            else:
                logger.warning("No se obtuvieron datos válidos")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error extrayendo datos: {e}")
            return pd.DataFrame()
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma los datos al formato requerido
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame transformado
        """
        try:
            # Hacer copia para no modificar original
            df_transformed = df.copy()
            
            # 1. Estandarizar nombres de columnas
            df_transformed = self._standardize_column_names(df_transformed)
            
            # 2. Procesar timestamp
            df_transformed = self._process_timestamp(df_transformed)
            
            # 3. Limpiar y validar contaminantes
            df_transformed = self._clean_pollutant_data(df_transformed)
            
            # 4. Procesar información de estaciones
            df_transformed = self._process_station_info(df_transformed)
            
            # 5. Agregar metadatos
            df_transformed = self._add_metadata(df_transformed)
            
            # 6. Filtrar registros válidos
            df_transformed = self._filter_valid_records(df_transformed)
            
            logger.info(f"Transformación completada: {len(df)} -> {len(df_transformed)} registros")
            return df_transformed
            
        except Exception as e:
            logger.error(f"Error en transformación: {e}")
            return pd.DataFrame()
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estandariza nombres de columnas"""
        column_mapping = {
            'ID': 'original_id',
            'ESTACION': 'station_id',
            'FECHA': 'date_raw',
            'HORA': 'hour_raw', 
            'LONGITUD': 'longitude',
            'LATITUD': 'latitude',
            'ALTITUD': 'altitude',
            'PM10': 'pm10',
            'PM2_5': 'pm25',
            'NO2': 'no2',
            'SO2': 'so2',
            'O3': 'o3',
            'CO': 'co',
            'DEPARTAMENTO': 'department',
            'PROVINCIA': 'province',
            'DISTRITO': 'district',
            'UBIGEO': 'ubigeo',
            'FECHA_CORTE': 'data_cut_date'
        }
        
        return df.rename(columns=column_mapping)
    
    def _process_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesa y crea timestamp unificado"""
        try:
            if 'date_raw' in df.columns and 'hour_raw' in df.columns:
                # Convertir fecha (formato YYYYMMDD)
                df['date'] = pd.to_datetime(df['date_raw'].astype(str), format='%Y%m%d', errors='coerce')
                
                # Convertir hora (formato HHMMSS a horas)
                df['hour'] = df['hour_raw'].apply(self._normalize_hour)
                
                # Crear timestamp combinado
                df['timestamp'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')
                
                # Eliminar columnas temporales
                df = df.drop(['date', 'hour'], axis=1, errors='ignore')
            
            return df
            
        except Exception as e:
            logger.error(f"Error procesando timestamp: {e}")
            return df
    
    def _normalize_hour(self, hour_value) -> int:
        """Normaliza hora a formato 0-23"""
        try:
            if pd.isna(hour_value):
                return 0
            
            hour_str = str(int(hour_value))
            
            # Formato HHMMSS -> extraer HH
            if len(hour_str) >= 6:
                return min(int(hour_str[:2]), 23)
            elif len(hour_str) >= 4:
                return min(int(hour_str[:2]), 23)
            else:
                return min(int(hour_str), 23)
                
        except:
            return 0
    
    def _clean_pollutant_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y valida datos de contaminantes"""
        pollutants = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']
        
        for pollutant in pollutants:
            if pollutant in df.columns:
                # Convertir a numérico
                df[pollutant] = pd.to_numeric(df[pollutant], errors='coerce')
                
                # Aplicar rangos de validación
                if pollutant in ['pm25', 'pm10', 'no2']:
                    min_val = VALIDATION_CONFIG.get(f'{pollutant}_min', 0)
                    max_val = VALIDATION_CONFIG.get(f'{pollutant}_max', 1000)
                    
                    # Marcar valores fuera de rango como NaN
                    mask = (df[pollutant] < min_val) | (df[pollutant] > max_val)
                    df.loc[mask, pollutant] = np.nan
                
                # Eliminar valores extremos (outliers) usando IQR
                if df[pollutant].notna().sum() > 100:  # Solo si hay suficientes datos
                    Q1 = df[pollutant].quantile(0.25)
                    Q3 = df[pollutant].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Límites para outliers
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    
                    outlier_mask = (df[pollutant] < lower_bound) | (df[pollutant] > upper_bound)
                    outliers_count = outlier_mask.sum()
                    
                    if outliers_count > 0:
                        logger.info(f"Eliminando {outliers_count} outliers de {pollutant}")
                        df.loc[outlier_mask, pollutant] = np.nan
        
        return df
    
    def _process_station_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesa información de estaciones"""
        # Crear nombre de estación si no existe
        if 'station_name' not in df.columns:
            df['station_name'] = df['station_id']
        
        # Limpiar nombres de estaciones
        if 'station_id' in df.columns:
            df['station_id'] = df['station_id'].str.strip().str.upper()
        
        # Limpiar nombres de distritos
        if 'district' in df.columns:
            df['district'] = df['district'].str.strip().str.upper()
        
        # Convertir coordenadas a numérico
        for coord in ['latitude', 'longitude', 'altitude']:
            if coord in df.columns:
                df[coord] = pd.to_numeric(df[coord], errors='coerce')
        
        return df
    
    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega metadatos al DataFrame"""
        # Agregar fuente de datos
        df['data_source'] = 'historical_file'
        
        # Agregar flag de calidad inicial
        df['quality_flag'] = 'valid'
        
        # Agregar timestamp de procesamiento
        df['processed_at'] = datetime.now()
        
        return df
    
    def _filter_valid_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra registros válidos"""
        initial_count = len(df)
        
        # Eliminar filas sin timestamp
        df = df[df['timestamp'].notna()]
        
        # Eliminar filas sin station_id
        df = df[df['station_id'].notna()]
        
        # Eliminar filas sin coordenadas (para análisis espacial)
        df = df[df['latitude'].notna() & df['longitude'].notna()]
        
        # Mantener solo filas con al menos un contaminante válido
        pollutant_cols = ['pm25', 'pm10', 'no2']
        df = df[df[pollutant_cols].notna().any(axis=1)]
        
        final_count = len(df)
        logger.info(f"Filtrado completado: {initial_count} -> {final_count} registros ({final_count/initial_count*100:.1f}% válidos)")
        
        return df
    
    def load_data(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Carga datos en la base de datos
        
        Args:
            df: DataFrame con datos transformados
            
        Returns:
            Estadísticas de carga
        """
        logger.info(f"Iniciando carga de {len(df)} registros")
        
        try:
            # Insertar mediciones
            insertion_stats = self.db_manager.insert_air_quality_data(df, source='etl_historical')
            
            # Actualizar información de estaciones
            stations_df = df[['station_id', 'station_name', 'district', 'latitude', 'longitude', 'altitude']].drop_duplicates()
            stations_updated = self.db_manager.update_stations(stations_df)
            
            insertion_stats['stations_updated'] = stations_updated
            
            logger.info(f"Carga completada - Insertados: {insertion_stats['inserted']}, Actualizados: {insertion_stats['updated']}, Estaciones: {stations_updated}")
            
            return insertion_stats
            
        except Exception as e:
            logger.error(f"Error en carga de datos: {e}")
            return {'inserted': 0, 'updated': 0, 'errors': 1, 'stations_updated': 0}
    
    def run_full_etl(self, file_path: str) -> Dict[str, Any]:
        """
        Ejecuta el proceso ETL completo
        
        Args:
            file_path: Ruta del archivo de datos
            
        Returns:
            Estadísticas del proceso
        """
        start_time = datetime.now()
        process_id = f"etl_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Iniciando ETL completo - Proceso ID: {process_id}")
        
        stats = {
            'process_id': process_id,
            'start_time': start_time,
            'file_path': file_path,
            'records_extracted': 0,
            'records_transformed': 0,
            'records_loaded': 0,
            'stations_updated': 0,
            'status': 'started'
        }
        
        try:
            # Extract
            df_extracted = self.extract_historical_data(file_path)
            stats['records_extracted'] = len(df_extracted)
            
            if len(df_extracted) == 0:
                stats['status'] = 'failed'
                stats['error'] = 'No data extracted'
                return stats
            
            # Transform (ya se hace en extract_historical_data)
            stats['records_transformed'] = len(df_extracted)
            
            # Load
            load_stats = self.load_data(df_extracted)
            stats['records_loaded'] = load_stats['inserted'] + load_stats['updated']
            stats['stations_updated'] = load_stats.get('stations_updated', 0)
            
            # Finalizar
            end_time = datetime.now()
            stats.update({
                'end_time': end_time,
                'duration_seconds': (end_time - start_time).total_seconds(),
                'status': 'completed'
            })
            
            logger.info(f"ETL completado exitosamente - {stats['records_loaded']} registros cargados")
            
        except Exception as e:
            error_msg = f"Error en ETL: {e}"
            logger.error(error_msg)
            stats.update({
                'status': 'failed',
                'error': error_msg,
                'end_time': datetime.now()
            })
        
        return stats

def main():
    """Función principal para ejecutar ETL"""
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ruta del archivo de datos históricos
    data_file = "/Users/juandiegogutierrezcortez/lima-air-dashboard/data/lima_air_quality_complete.csv"
    
    if not os.path.exists(data_file):
        logger.error(f"Archivo de datos no encontrado: {data_file}")
        return
    
    # Ejecutar ETL
    etl = AirQualityETL(use_sqlite=True)
    
    # Probar conexión
    if not etl.db_manager.test_connection():
        logger.error("No se pudo conectar a la base de datos")
        return
    
    # Ejecutar proceso
    stats = etl.run_full_etl(data_file)
    
    # Mostrar resultados
    print(f"\n{'='*60}")
    print("RESUMEN DE PROCESO ETL")
    print(f"{'='*60}")
    print(f"Proceso ID: {stats['process_id']}")
    print(f"Estado: {stats['status']}")
    print(f"Archivo: {stats['file_path']}")
    print(f"Duración: {stats.get('duration_seconds', 0):.2f} segundos")
    print(f"Registros extraídos: {stats['records_extracted']:,}")
    print(f"Registros transformados: {stats['records_transformed']:,}")
    print(f"Registros cargados: {stats['records_loaded']:,}")
    print(f"Estaciones actualizadas: {stats['stations_updated']}")
    
    if stats['status'] == 'failed':
        print(f"Error: {stats.get('error', 'Error desconocido')}")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
