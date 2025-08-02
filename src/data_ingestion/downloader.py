"""
Descarga automática de datos de calidad del aire desde múltiples fuentes
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import requests
import zipfile
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import time
import hashlib

from config.settings import DATA_SOURCES, DOWNLOAD_CONFIG, LOG_CONFIG
from src.database.connection import get_db_manager

# Configurar logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG['level']),
    format=LOG_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOG_CONFIG['file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataDownloader:
    """
    Descargador automático de datos de calidad del aire
    """
    
    def __init__(self, data_dir: str = None):
        """
        Inicializa el descargador
        
        Args:
            data_dir: Directorio para guardar datos descargados
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        self.downloads_dir = os.path.join(self.data_dir, 'downloads')
        self.backups_dir = os.path.join(self.data_dir, 'backups')
        
        # Crear directorios
        os.makedirs(self.downloads_dir, exist_ok=True)
        os.makedirs(self.backups_dir, exist_ok=True)
        
        self.db_manager = get_db_manager(use_sqlite=True)  # Usar SQLite por defecto para desarrollo
        
    def download_from_datos_abiertos(self) -> Optional[str]:
        """
        Descarga datos desde el portal de Datos Abiertos del Gobierno Peruano
        
        Returns:
            Ruta del archivo descargado o None si falló
        """
        source_config = DATA_SOURCES['datos_abiertos_pe']
        logger.info(f"Iniciando descarga desde: {source_config['description']}")
        
        try:
            # URL del CSV (esta URL puede cambiar, necesitará actualizarse)
            csv_url = source_config.get('csv_url')
            if not csv_url:
                logger.warning("URL del CSV no configurada para datos abiertos")
                return None
            
            # Nombre del archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"datos_abiertos_{timestamp}.csv"
            filepath = os.path.join(self.downloads_dir, filename)
            
            # Descargar archivo
            response = requests.get(
                csv_url,
                timeout=DOWNLOAD_CONFIG['timeout_seconds'],
                stream=True
            )
            response.raise_for_status()
            
            # Guardar archivo
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verificar que el archivo se descargó correctamente
            if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:  # Al menos 1KB
                logger.info(f"Archivo descargado exitosamente: {filepath}")
                return filepath
            else:
                logger.error("Archivo descargado está vacío o corrupto")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Error descargando desde datos abiertos: {e}")
            return None
        except Exception as e:
            logger.error(f"Error inesperado en descarga: {e}")
            return None
    
    def download_from_openaq(self, hours_back: int = 24) -> Optional[pd.DataFrame]:
        """
        Descarga datos recientes desde OpenAQ API
        
        Args:
            hours_back: Horas hacia atrás para obtener datos
            
        Returns:
            DataFrame con datos o None si falló
        """
        source_config = DATA_SOURCES['openaq']
        api_key = source_config.get('api_key')
        
        if not api_key:
            logger.warning("API Key de OpenAQ no configurada")
            return None
        
        logger.info(f"Descargando datos de OpenAQ (últimas {hours_back} horas)")
        
        try:
            # Calcular timestamps
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours_back)
            
            # Parámetros para la API
            params = {
                'parameter': 'pm25',  # Comenzar con PM2.5
                'location': 'PE',     # Perú
                'date_from': start_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'date_to': end_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'limit': 10000
            }
            
            headers = {'X-API-Key': api_key}
            
            # Llamar a la API
            response = requests.get(
                f"{source_config['base_url']}/measurements",
                params=params,
                headers=headers,
                timeout=DOWNLOAD_CONFIG['timeout_seconds']
            )
            response.raise_for_status()
            
            data = response.json()
            measurements = data.get('results', [])
            
            if not measurements:
                logger.warning("No se obtuvieron datos de OpenAQ")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(measurements)
            
            # Filtrar por bounding box de Lima
            if 'coordinates' in df.columns:
                df = df[df['coordinates'].notna()]
                df['latitude'] = df['coordinates'].apply(lambda x: x.get('latitude') if isinstance(x, dict) else None)
                df['longitude'] = df['coordinates'].apply(lambda x: x.get('longitude') if isinstance(x, dict) else None)
                
                # Filtro geográfico para Lima
                lima_mask = (
                    (df['latitude'] >= -12.30) & (df['latitude'] <= -11.80) &
                    (df['longitude'] >= -77.30) & (df['longitude'] <= -76.90)
                )
                df = df[lima_mask]
            
            if len(df) > 0:
                logger.info(f"Obtenidos {len(df)} registros de OpenAQ para Lima")
                return df
            else:
                logger.warning("No se encontraron datos de Lima en OpenAQ")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Error en API de OpenAQ: {e}")
            return None
        except Exception as e:
            logger.error(f"Error procesando datos de OpenAQ: {e}")
            return None
    
    def process_csv_file(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Procesa un archivo CSV descargado
        
        Args:
            filepath: Ruta del archivo CSV
            
        Returns:
            DataFrame procesado o None si falló
        """
        try:
            logger.info(f"Procesando archivo: {filepath}")
            
            # Leer CSV
            df = pd.read_csv(filepath)
            logger.info(f"Archivo leído: {len(df)} registros, {len(df.columns)} columnas")
            
            # Mapear columnas al esquema estándar
            df = self._standardize_columns(df)
            
            # Validar y limpiar datos
            df = self._validate_and_clean_data(df)
            
            if len(df) > 0:
                logger.info(f"Datos procesados: {len(df)} registros válidos")
                return df
            else:
                logger.warning("No quedaron registros válidos después del procesamiento")
                return None
                
        except Exception as e:
            logger.error(f"Error procesando archivo CSV: {e}")
            return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estandariza nombres de columnas al esquema de la base de datos
        """
        # Mapeo de columnas conocidas
        column_mapping = {
            'ESTACION': 'station_id',
            'DISTRITO': 'district',
            'LATITUD': 'latitude',
            'LONGITUD': 'longitude',
            'ALTITUD': 'altitude',
            'FECHA': 'date',
            'HORA': 'hour',
            'PM2_5': 'pm25',
            'PM10': 'pm10',
            'NO2': 'no2',
            'SO2': 'so2',
            'O3': 'o3',
            'CO': 'co'
        }
        
        # Aplicar mapeo
        df = df.rename(columns=column_mapping)
        
        # Crear timestamp combinando fecha y hora si están separadas
        if 'date' in df.columns and 'hour' in df.columns:
            try:
                # Convertir fecha (formato YYYYMMDD) y hora (formato HHMMSS)
                df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce')
                
                # Convertir hora (puede estar en formato HHMMSS o HH)
                df['hour_normalized'] = df['hour'].apply(self._normalize_hour)
                
                # Combinar fecha y hora
                df['timestamp'] = df['date'] + pd.to_timedelta(df['hour_normalized'], unit='h')
                
                # Eliminar columnas temporales
                df = df.drop(['date', 'hour', 'hour_normalized'], axis=1)
                
            except Exception as e:
                logger.warning(f"Error procesando timestamp: {e}")
        
        # Asegurar que station_name existe
        if 'station_id' in df.columns and 'station_name' not in df.columns:
            df['station_name'] = df['station_id']
        
        return df
    
    def _normalize_hour(self, hour_value) -> int:
        """
        Normaliza valores de hora a formato de 24 horas
        """
        try:
            if pd.isna(hour_value):
                return 0
            
            hour_str = str(int(hour_value))
            
            # Si está en formato HHMMSS, extraer solo HH
            if len(hour_str) >= 6:
                return int(hour_str[:2])
            # Si está en formato HH0000, extraer HH
            elif len(hour_str) >= 4:
                return int(hour_str[:2])
            # Si está en formato HH
            else:
                return min(int(hour_str), 23)
                
        except:
            return 0
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valida y limpia los datos según configuración
        """
        from config.settings import VALIDATION_CONFIG
        
        initial_count = len(df)
        
        # Eliminar filas sin timestamp
        if 'timestamp' in df.columns:
            df = df[df['timestamp'].notna()]
        
        # Eliminar filas sin station_id
        if 'station_id' in df.columns:
            df = df[df['station_id'].notna()]
        
        # Validar rangos de contaminantes
        for pollutant in ['pm25', 'pm10', 'no2']:
            if pollutant in df.columns:
                min_val = VALIDATION_CONFIG.get(f'{pollutant}_min', 0)
                max_val = VALIDATION_CONFIG.get(f'{pollutant}_max', 1000)
                
                # Filtrar valores fuera de rango
                valid_mask = (
                    df[pollutant].isna() |
                    ((df[pollutant] >= min_val) & (df[pollutant] <= max_val))
                )
                df = df[valid_mask]
        
        final_count = len(df)
        logger.info(f"Validación completada: {initial_count} -> {final_count} registros ({final_count/initial_count*100:.1f}% válidos)")
        
        return df
    
    def run_hourly_ingestion(self) -> Dict[str, Any]:
        """
        Ejecuta el proceso completo de ingesta horaria
        
        Returns:
            Estadísticas del proceso
        """
        process_id = f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"Iniciando ingesta horaria - Proceso ID: {process_id}")
        
        stats = {
            'process_id': process_id,
            'start_time': start_time,
            'sources_processed': 0,
            'total_records_inserted': 0,
            'total_records_updated': 0,
            'errors': []
        }
        
        try:
            # Registrar inicio del proceso
            self.db_manager.log_ingestion_process(
                process_id=process_id,
                source='automated',
                status='started',
                start_time=start_time
            )
            
            # 1. Intentar descarga desde OpenAQ (datos recientes)
            try:
                openaq_data = self.download_from_openaq(hours_back=2)
                if openaq_data is not None:
                    # Procesar datos de OpenAQ
                    insertion_stats = self.db_manager.insert_air_quality_data(
                        openaq_data, source='openaq'
                    )
                    stats['total_records_inserted'] += insertion_stats['inserted']
                    stats['total_records_updated'] += insertion_stats['updated']
                    stats['sources_processed'] += 1
                    logger.info("Datos de OpenAQ procesados exitosamente")
                
            except Exception as e:
                error_msg = f"Error procesando OpenAQ: {e}"
                logger.error(error_msg)
                stats['errors'].append(error_msg)
            
            # 2. Procesar archivo local existente si no hay datos recientes
            local_file = os.path.join(self.data_dir, 'lima_air_quality_complete.csv')
            if os.path.exists(local_file):
                try:
                    # Verificar si hay datos nuevos que procesar
                    latest_db_time = self.db_manager.get_latest_measurement_time()
                    
                    # Solo procesar si no hay datos recientes en la DB
                    if latest_db_time is None or (datetime.now() - latest_db_time).days > 1:
                        logger.info("Procesando archivo local de datos históricos")
                        
                        # Leer y procesar muestra del archivo (para evitar sobrecarga)
                        df_sample = pd.read_csv(local_file, nrows=10000)
                        df_processed = self._standardize_columns(df_sample)
                        df_processed = self._validate_and_clean_data(df_processed)
                        
                        if len(df_processed) > 0:
                            insertion_stats = self.db_manager.insert_air_quality_data(
                                df_processed, source='historical'
                            )
                            stats['total_records_inserted'] += insertion_stats['inserted']
                            stats['total_records_updated'] += insertion_stats['updated']
                            stats['sources_processed'] += 1
                
                except Exception as e:
                    error_msg = f"Error procesando archivo local: {e}"
                    logger.error(error_msg)
                    stats['errors'].append(error_msg)
            
            # Calcular estadísticas finales
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            stats.update({
                'end_time': end_time,
                'duration_seconds': duration,
                'status': 'completed' if len(stats['errors']) == 0 else 'completed_with_errors'
            })
            
            # Registrar finalización del proceso
            self.db_manager.log_ingestion_process(
                process_id=process_id,
                source='automated',
                status=stats['status'],
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                records_inserted=stats['total_records_inserted'],
                records_updated=stats['total_records_updated']
            )
            
            logger.info(f"Ingesta completada - Insertados: {stats['total_records_inserted']}, Actualizados: {stats['total_records_updated']}")
            
        except Exception as e:
            error_msg = f"Error crítico en ingesta: {e}"
            logger.error(error_msg)
            stats['errors'].append(error_msg)
            stats['status'] = 'failed'
            
            # Registrar fallo
            self.db_manager.log_ingestion_process(
                process_id=process_id,
                source='automated',
                status='failed',
                start_time=start_time,
                end_time=datetime.now(),
                error_message=error_msg
            )
        
        return stats

def main():
    """Función principal para ejecutar la ingesta"""
    downloader = DataDownloader()
    
    # Probar conexión a base de datos
    if not downloader.db_manager.test_connection():
        logger.error("No se pudo conectar a la base de datos")
        return
    
    # Ejecutar ingesta
    stats = downloader.run_hourly_ingestion()
    
    print(f"\n{'='*50}")
    print("RESUMEN DE INGESTA")
    print(f"{'='*50}")
    print(f"Proceso ID: {stats['process_id']}")
    print(f"Estado: {stats['status']}")
    print(f"Duración: {stats.get('duration_seconds', 0):.2f} segundos")
    print(f"Fuentes procesadas: {stats['sources_processed']}")
    print(f"Registros insertados: {stats['total_records_inserted']}")
    print(f"Registros actualizados: {stats['total_records_updated']}")
    
    if stats['errors']:
        print(f"\nErrores ({len(stats['errors'])}):")
        for error in stats['errors']:
            print(f"  - {error}")
    
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
