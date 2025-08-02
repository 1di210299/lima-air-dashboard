"""
Esquemas de base de datos para el dashboard de calidad del aire de Lima
"""

import sqlalchemy as sa
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class AirQualityMeasurement(Base):
    """
    Tabla principal de mediciones de calidad del aire
    """
    __tablename__ = 'air_quality_measurements'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    station_id = Column(String(50), nullable=False, index=True)
    station_name = Column(String(100), nullable=False)
    district = Column(String(50), nullable=False, index=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    altitude = Column(Float, nullable=True)
    
    # Timestamp de la medición
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Contaminantes principales
    pm25 = Column(Float, nullable=True)  # µg/m³
    pm10 = Column(Float, nullable=True)  # µg/m³
    no2 = Column(Float, nullable=True)   # µg/m³
    
    # Contaminantes adicionales
    so2 = Column(Float, nullable=True)   # µg/m³
    o3 = Column(Float, nullable=True)    # µg/m³
    co = Column(Float, nullable=True)    # mg/m³
    
    # Metadatos
    data_source = Column(String(50), nullable=False, default='manual')
    quality_flag = Column(String(20), nullable=True)  # 'valid', 'suspect', 'invalid'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Índices compuestos para consultas eficientes
    __table_args__ = (
        Index('idx_station_timestamp', 'station_id', 'timestamp'),
        Index('idx_district_timestamp', 'district', 'timestamp'),
        Index('idx_timestamp_pm25', 'timestamp', 'pm25'),
    )

class Station(Base):
    """
    Tabla de estaciones de monitoreo
    """
    __tablename__ = 'stations'
    
    station_id = Column(String(50), primary_key=True)
    station_name = Column(String(100), nullable=False)
    district = Column(String(50), nullable=False)
    province = Column(String(50), nullable=False, default='LIMA')
    department = Column(String(50), nullable=False, default='LIMA')
    
    # Coordenadas
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    altitude = Column(Float, nullable=True)
    
    # Estado de la estación
    is_active = Column(Boolean, default=True)
    installation_date = Column(DateTime, nullable=True)
    last_measurement = Column(DateTime, nullable=True)
    
    # Metadatos
    description = Column(Text, nullable=True)
    operator = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DataIngestionLog(Base):
    """
    Log de procesos de ingesta de datos
    """
    __tablename__ = 'data_ingestion_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    process_id = Column(String(50), nullable=False)
    source = Column(String(50), nullable=False)  # 'datos_abiertos', 'openaq', etc.
    status = Column(String(20), nullable=False)  # 'started', 'completed', 'failed'
    
    # Estadísticas del proceso
    records_processed = Column(Integer, default=0)
    records_inserted = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_skipped = Column(Integer, default=0)
    
    # Timestamps
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # Detalles del proceso
    file_processed = Column(String(255), nullable=True)
    error_message = Column(Text, nullable=True)
    details = Column(Text, nullable=True)  # JSON con detalles adicionales
    
    created_at = Column(DateTime, default=datetime.utcnow)

class AirQualityTimeSeries(Base):
    """
    Tabla agregada por horas para series temporales y modelado
    """
    __tablename__ = 'air_quality_timeseries'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    district = Column(String(50), nullable=False, index=True)
    hour = Column(DateTime, nullable=False, index=True)  # Redondeado a la hora
    
    # Promedios por hora
    pm25_avg = Column(Float, nullable=True)
    pm25_min = Column(Float, nullable=True)
    pm25_max = Column(Float, nullable=True)
    pm25_count = Column(Integer, default=0)
    
    pm10_avg = Column(Float, nullable=True)
    pm10_min = Column(Float, nullable=True)
    pm10_max = Column(Float, nullable=True)
    pm10_count = Column(Integer, default=0)
    
    no2_avg = Column(Float, nullable=True)
    no2_min = Column(Float, nullable=True)
    no2_max = Column(Float, nullable=True)
    no2_count = Column(Integer, default=0)
    
    # Metadatos
    stations_reporting = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_district_hour', 'district', 'hour'),
    )

# SQL para crear las tablas (útil para deployments)
CREATE_TABLES_SQL = """
-- Crear base de datos (ejecutar como superuser)
-- CREATE DATABASE lima_air_quality;

-- Crear extensiones útiles
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Las tablas se crean automáticamente con SQLAlchemy Base.metadata.create_all()
"""

# Funciones útiles para la base de datos
def get_table_names():
    """Retorna los nombres de todas las tablas"""
    return [table.name for table in Base.metadata.tables.values()]

def get_create_sql():
    """Retorna el SQL para crear todas las tablas"""
    from sqlalchemy import create_engine
    from sqlalchemy.schema import CreateTable
    
    # Motor temporal para generar SQL
    engine = create_engine('postgresql://user:pass@localhost/db', echo=True)
    
    sql_statements = []
    for table in Base.metadata.tables.values():
        sql_statements.append(str(CreateTable(table).compile(engine)))
    
    return '\n\n'.join(sql_statements)
