"""
Conexión y operaciones de base de datos
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
import pandas as pd
from typing import Optional, List, Dict, Any

from config.settings import DATABASE_URL, SQLITE_DATABASE, DATABASE_CONFIG
from src.database.models import Base, AirQualityMeasurement, Station, DataIngestionLog, AirQualityTimeSeries

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Gestor de conexiones y operaciones de base de datos
    """
    
    def __init__(self, use_sqlite: bool = False):
        """
        Inicializa el gestor de base de datos
        
        Args:
            use_sqlite: Si usar SQLite en lugar de PostgreSQL (para desarrollo)
        """
        self.use_sqlite = use_sqlite
        
        if use_sqlite:
            # Crear directorio de datos si no existe
            os.makedirs(os.path.dirname(SQLITE_DATABASE), exist_ok=True)
            self.database_url = f"sqlite:///{SQLITE_DATABASE}"
        else:
            self.database_url = DATABASE_URL
        
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Inicializa la conexión a la base de datos"""
        try:
            # Crear engine
            if self.use_sqlite:
                self.engine = create_engine(
                    self.database_url,
                    echo=False,
                    connect_args={"check_same_thread": False}
                )
            else:
                self.engine = create_engine(
                    self.database_url,
                    echo=False,
                    pool_size=10,
                    max_overflow=20
                )
            
            # Crear session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Crear tablas
            self.create_tables()
            
            logger.info(f"Base de datos inicializada: {'SQLite' if self.use_sqlite else 'PostgreSQL'}")
            
        except Exception as e:
            logger.error(f"Error inicializando base de datos: {e}")
            raise
    
    def create_tables(self):
        """Crea todas las tablas si no existen"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Tablas creadas exitosamente")
        except Exception as e:
            logger.error(f"Error creando tablas: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Context manager para sesiones de base de datos"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error en sesión de base de datos: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Prueba la conexión a la base de datos"""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            logger.info("Conexión a base de datos exitosa")
            return True
        except Exception as e:
            logger.error(f"Error en conexión a base de datos: {e}")
            return False
    
    def insert_air_quality_data(self, df: pd.DataFrame, source: str = 'manual') -> Dict[str, int]:
        """
        Inserta datos de calidad del aire desde un DataFrame
        
        Args:
            df: DataFrame con datos de calidad del aire
            source: Fuente de los datos
            
        Returns:
            Dict con estadísticas de inserción
        """
        stats = {'inserted': 0, 'updated': 0, 'skipped': 0, 'errors': 0}
        
        try:
            with self.get_session() as session:
                for _, row in df.iterrows():
                    try:
                        # Verificar si el registro ya existe
                        existing = session.query(AirQualityMeasurement).filter_by(
                            station_id=row.get('station_id'),
                            timestamp=row.get('timestamp')
                        ).first()
                        
                        if existing:
                            # Actualizar registro existente
                            for col in ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']:
                                if col in row and pd.notna(row[col]):
                                    setattr(existing, col, float(row[col]))
                            
                            existing.data_source = source
                            existing.updated_at = pd.Timestamp.now()
                            stats['updated'] += 1
                        else:
                            # Crear nuevo registro
                            measurement = AirQualityMeasurement(
                                station_id=row.get('station_id'),
                                station_name=row.get('station_name', row.get('station_id')),
                                district=row.get('district'),
                                latitude=float(row['latitude']) if pd.notna(row.get('latitude')) else None,
                                longitude=float(row['longitude']) if pd.notna(row.get('longitude')) else None,
                                altitude=float(row['altitude']) if pd.notna(row.get('altitude')) else None,
                                timestamp=row.get('timestamp'),
                                pm25=float(row['pm25']) if pd.notna(row.get('pm25')) else None,
                                pm10=float(row['pm10']) if pd.notna(row.get('pm10')) else None,
                                no2=float(row['no2']) if pd.notna(row.get('no2')) else None,
                                so2=float(row['so2']) if pd.notna(row.get('so2')) else None,
                                o3=float(row['o3']) if pd.notna(row.get('o3')) else None,
                                co=float(row['co']) if pd.notna(row.get('co')) else None,
                                data_source=source,
                                quality_flag='valid'
                            )
                            session.add(measurement)
                            stats['inserted'] += 1
                    
                    except Exception as e:
                        logger.error(f"Error procesando fila: {e}")
                        stats['errors'] += 1
                        continue
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Error insertando datos: {e}")
            raise
        
        logger.info(f"Datos insertados - Inserted: {stats['inserted']}, Updated: {stats['updated']}, Errors: {stats['errors']}")
        return stats
    
    def update_stations(self, df: pd.DataFrame) -> int:
        """
        Actualiza la tabla de estaciones
        
        Args:
            df: DataFrame con información de estaciones
            
        Returns:
            Número de estaciones actualizadas
        """
        count = 0
        
        try:
            with self.get_session() as session:
                for _, row in df.iterrows():
                    station = session.query(Station).filter_by(
                        station_id=row.get('station_id')
                    ).first()
                    
                    if station:
                        # Actualizar estación existente
                        station.station_name = row.get('station_name', station.station_name)
                        station.district = row.get('district', station.district)
                        station.latitude = float(row['latitude']) if pd.notna(row.get('latitude')) else station.latitude
                        station.longitude = float(row['longitude']) if pd.notna(row.get('longitude')) else station.longitude
                        station.altitude = float(row['altitude']) if pd.notna(row.get('altitude')) else station.altitude
                        station.updated_at = pd.Timestamp.now()
                    else:
                        # Crear nueva estación
                        station = Station(
                            station_id=row.get('station_id'),
                            station_name=row.get('station_name', row.get('station_id')),
                            district=row.get('district'),
                            latitude=float(row['latitude']) if pd.notna(row.get('latitude')) else None,
                            longitude=float(row['longitude']) if pd.notna(row.get('longitude')) else None,
                            altitude=float(row['altitude']) if pd.notna(row.get('altitude')) else None,
                            is_active=True
                        )
                        session.add(station)
                    
                    count += 1
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Error actualizando estaciones: {e}")
            raise
        
        logger.info(f"Estaciones actualizadas: {count}")
        return count
    
    def log_ingestion_process(self, process_id: str, source: str, status: str, **kwargs) -> int:
        """
        Registra un proceso de ingesta en el log
        
        Args:
            process_id: ID único del proceso
            source: Fuente de datos
            status: Estado del proceso
            **kwargs: Parámetros adicionales
            
        Returns:
            ID del log creado
        """
        try:
            with self.get_session() as session:
                log_entry = DataIngestionLog(
                    process_id=process_id,
                    source=source,
                    status=status,
                    **kwargs
                )
                session.add(log_entry)
                session.commit()
                return log_entry.id
                
        except Exception as e:
            logger.error(f"Error registrando proceso: {e}")
            raise
    
    def get_latest_measurement_time(self, station_id: Optional[str] = None) -> Optional[pd.Timestamp]:
        """
        Obtiene el timestamp de la última medición
        
        Args:
            station_id: ID de estación específica (opcional)
            
        Returns:
            Timestamp de la última medición
        """
        try:
            with self.get_session() as session:
                query = session.query(AirQualityMeasurement.timestamp)
                
                if station_id:
                    query = query.filter(AirQualityMeasurement.station_id == station_id)
                
                result = query.order_by(AirQualityMeasurement.timestamp.desc()).first()
                
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"Error obteniendo última medición: {e}")
            return None
    
    def get_stations(self) -> List[Dict[str, Any]]:
        """
        Obtiene lista de todas las estaciones activas
        
        Returns:
            Lista de estaciones
        """
        try:
            with self.get_session() as session:
                stations = session.query(Station).filter(Station.is_active == True).all()
                
                return [{
                    'station_id': s.station_id,
                    'station_name': s.station_name,
                    'district': s.district,
                    'latitude': s.latitude,
                    'longitude': s.longitude,
                    'altitude': s.altitude
                } for s in stations]
                
        except Exception as e:
            logger.error(f"Error obteniendo estaciones: {e}")
            return []

# Instancia global del gestor de base de datos
db_manager = None

def get_db_manager(use_sqlite: bool = False) -> DatabaseManager:
    """
    Obtiene o crea el gestor de base de datos
    
    Args:
        use_sqlite: Si usar SQLite en lugar de PostgreSQL
        
    Returns:
        Instancia del DatabaseManager
    """
    global db_manager
    
    if db_manager is None:
        db_manager = DatabaseManager(use_sqlite=use_sqlite)
    
    return db_manager
