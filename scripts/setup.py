"""
Script de inicialización para configurar el entorno del dashboard de calidad del aire
"""

import os
import sys
from pathlib import Path

# Agregar el directorio raíz al path de Python
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configurar variables de entorno por defecto
def setup_environment():
    """Configura variables de entorno por defecto"""
    env_file = project_root / '.env'
    
    if not env_file.exists():
        print("⚠️  Archivo .env no encontrado. Creando archivo de ejemplo...")
        
        env_content = """# Lima Air Quality Dashboard - Environment Variables

# Database Configuration (PostgreSQL)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=lima_air_quality
DB_USER=postgres
DB_PASSWORD=your_password_here

# OpenAQ API (for real-time data)
OPENAQ_API_KEY=your_openaq_api_key_here

# Data Sources
DATA_ABIERTOS_URL=https://datosabiertos.gob.pe/sites/default/files/datos_horarios_contaminacion_lima.csv

# Logging
LOG_LEVEL=INFO

# Development settings
DEVELOPMENT=true
USE_SQLITE=true

# Future: Twilio (for notifications)
# TWILIO_ACCOUNT_SID=your_account_sid
# TWILIO_AUTH_TOKEN=your_auth_token
# TWILIO_PHONE_NUMBER=+1234567890

# Future: API Settings
# API_SECRET_KEY=your_secret_key_here
# API_HOST=0.0.0.0
# API_PORT=8000
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"✅ Archivo .env creado en: {env_file}")
        print("📝 Por favor, edita el archivo .env con tus credenciales antes de continuar.")
        return False
    
    return True

def create_directories():
    """Crea directorios necesarios para el proyecto"""
    directories = [
        'data/downloads',
        'data/backups',
        'logs',
        'models',
        'src/api',
        'src/ml',
        'src/dashboard',
        'src/notifications',
        'tests',
        'docs'
    ]
    
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Directorios del proyecto creados")

def create_init_files():
    """Crea archivos __init__.py necesarios"""
    init_files = [
        'src/__init__.py',
        'src/data_ingestion/__init__.py',
        'src/database/__init__.py',
        'src/api/__init__.py',
        'src/ml/__init__.py',
        'src/dashboard/__init__.py',
        'src/notifications/__init__.py',
        'config/__init__.py',
        'tests/__init__.py'
    ]
    
    for init_file in init_files:
        file_path = project_root / init_file
        if not file_path.exists():
            file_path.touch()
    
    print("✅ Archivos __init__.py creados")

def check_dependencies():
    """Verifica si las dependencias están instaladas"""
    try:
        import pandas
        import sqlalchemy
        import requests
        print("✅ Dependencias principales encontradas")
        return True
    except ImportError as e:
        print(f"❌ Dependencia faltante: {e}")
        print("💡 Ejecuta: pip install -r requirements.txt")
        return False

def initialize_database():
    """Inicializa la base de datos (SQLite para desarrollo)"""
    try:
        from src.database.connection import get_db_manager
        
        # Usar SQLite para desarrollo inicial
        db_manager = get_db_manager(use_sqlite=True)
        
        if db_manager.test_connection():
            print("✅ Base de datos SQLite inicializada")
            return True
        else:
            print("❌ Error conectando a la base de datos")
            return False
            
    except Exception as e:
        print(f"❌ Error inicializando base de datos: {e}")
        return False

def run_sample_etl():
    """Ejecuta una muestra del proceso ETL"""
    try:
        from src.data_ingestion.etl import AirQualityETL
        
        data_file = project_root / 'data' / 'lima_air_quality_complete.csv'
        
        if not data_file.exists():
            print(f"⚠️  Archivo de datos no encontrado: {data_file}")
            print("💡 Asegúrate de tener el archivo lima_air_quality_complete.csv en el directorio data/")
            return False
        
        print("🔄 Ejecutando muestra del proceso ETL...")
        etl = AirQualityETL(use_sqlite=True)
        
        # Procesar solo una muestra pequeña
        import pandas as pd
        df_sample = pd.read_csv(data_file, nrows=1000)
        transformed_df = etl.transform_data(df_sample)
        
        if len(transformed_df) > 0:
            load_stats = etl.load_data(transformed_df)
            print(f"✅ ETL de muestra completado: {load_stats['inserted']} registros insertados")
            return True
        else:
            print("❌ No se pudieron transformar los datos")
            return False
            
    except Exception as e:
        print(f"❌ Error en ETL de muestra: {e}")
        return False

def main():
    """Función principal de inicialización"""
    print("🚀 INICIALIZANDO LIMA AIR QUALITY DASHBOARD")
    print("=" * 50)
    
    success = True
    
    # 1. Configurar entorno
    print("\n1️⃣ Configurando entorno...")
    if not setup_environment():
        success = False
    
    # 2. Crear directorios
    print("\n2️⃣ Creando estructura de directorios...")
    create_directories()
    
    # 3. Crear archivos __init__.py
    print("\n3️⃣ Creando archivos de inicialización...")
    create_init_files()
    
    # 4. Verificar dependencias
    print("\n4️⃣ Verificando dependencias...")
    if not check_dependencies():
        success = False
    
    # 5. Inicializar base de datos
    if success:
        print("\n5️⃣ Inicializando base de datos...")
        if not initialize_database():
            success = False
    
    # 6. Ejecutar ETL de muestra
    if success:
        print("\n6️⃣ Ejecutando ETL de muestra...")
        run_sample_etl()
    
    # Resumen final
    print(f"\n{'=' * 50}")
    if success:
        print("✅ INICIALIZACIÓN COMPLETADA")
        print("\n📋 PRÓXIMOS PASOS:")
        print("1. Edita el archivo .env con tus credenciales")
        print("2. Ejecuta: python src/data_ingestion/etl.py")
        print("3. Ejecuta: python src/data_ingestion/downloader.py")
        print("4. ¡Listo para la Fase 2!")
    else:
        print("❌ INICIALIZACIÓN CON ERRORES")
        print("\n🔧 ACCIONES REQUERIDAS:")
        print("1. Instala dependencias: pip install -r requirements.txt")
        print("2. Configura variables de entorno en .env")
        print("3. Ejecuta este script nuevamente")
    
    print(f"{'=' * 50}\n")

if __name__ == "__main__":
    main()
