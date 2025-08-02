"""
Configuración de la base de datos PostgreSQL para el dashboard de calidad del aire
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Configuración de base de datos
DATABASE_CONFIG = {
    'development': {
        'url': 'sqlite:///lima_air_quality.db',
        'echo': True
    },
    'production': {
        'url': 'postgresql://user:password@localhost:5432/lima_air_quality',
        'echo': False
    }
}

# Configuración actual del entorno
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# URLs de base de datos
DATABASE_URL = DATABASE_CONFIG[ENVIRONMENT]['url']

# Configuración de SQLite para desarrollo/testing
SQLITE_DATABASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'lima_air_quality.db')

# URLs de datos abiertos
DATA_SOURCES = {
    'datos_abiertos_pe': {
        'url': 'https://datosabiertos.gob.pe/dataset/monitoreo-de-los-contaminantes-del-aire-en-lima-metropolitana-servicio-nacional-de-0',
        'csv_url': 'https://datosabiertos.gob.pe/sites/default/files/datos_horarios_contaminacion_lima.csv',
        'description': 'Portal de Datos Abiertos del Gobierno Peruano'
    },
    'openaq': {
        'base_url': 'https://api.openaq.org/v2',
        'api_key': os.getenv('OPENAQ_API_KEY'),
        'description': 'OpenAQ API para datos en tiempo real'
    }
}

# Límites de contaminantes para validación (µg/m³)
POLLUTANT_LIMITS = {
    'pm25': {'min': 0, 'max': 1000, 'who_guideline': 15, 'alert_threshold': 75},
    'pm10': {'min': 0, 'max': 2000, 'who_guideline': 45, 'alert_threshold': 150},
    'so2': {'min': 0, 'max': 1000, 'who_guideline': 40, 'alert_threshold': 200},
    'no2': {'min': 0, 'max': 500, 'who_guideline': 25, 'alert_threshold': 100},
    'o3': {'min': 0, 'max': 400, 'who_guideline': 100, 'alert_threshold': 180},
    'co': {'min': 0, 'max': 50000, 'who_guideline': 4000, 'alert_threshold': 15000}
}

# Reglas de validación
VALIDATION_RULES = {
    'lima_coordinates': {
        'lat_min': -12.5,
        'lat_max': -11.5,
        'lon_min': -77.5,
        'lon_max': -76.5
    },
    'temporal_range': {
        'min_years_back': 20,
        'max_days_future': 1
    },
    'data_quality': {
        'min_completeness': 0.3,  # 30% mínimo de datos para considerar válida una medición
        'max_consecutive_missing': 72,  # Máximo 72 horas consecutivas sin datos
        'outlier_threshold': 3  # Múltiplo de IQR para considerar outlier
    }
}

# Configuración de logging
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'ingestion.log')
}

# Configuración de intervalos de descarga
DOWNLOAD_CONFIG = {
    'interval_hours': 1,  # Cada hora
    'max_retries': 3,
    'timeout_seconds': 300,  # 5 minutos
    'backup_days': 30  # Mantener backups por 30 días
}

# Configuración de validación de datos
VALIDATION_CONFIG = {
    'pm25_min': 0,
    'pm25_max': 1000,  # µg/m³
    'pm10_min': 0,
    'pm10_max': 2000,  # µg/m³
    'no2_min': 0,
    'no2_max': 500,    # µg/m³
    'required_columns': ['station_id', 'district', 'timestamp', 'pm25', 'pm10', 'no2']
}
