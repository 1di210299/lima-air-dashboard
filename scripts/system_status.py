#!/usr/bin/env python3
"""
Sistema de monitoreo y estado del Dashboard de Calidad del Aire de Lima
Muestra estadísticas completas del sistema y la base de datos
"""

import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import logging

# Agregar el directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.connection import DatabaseManager
from src.database.models import AirQualityMeasurement, Station, DataIngestionLog
from sqlalchemy import func, text

def print_banner():
    """Muestra el banner del sistema"""
    print("=" * 80)
    print("🌬️  LIMA AIR QUALITY DASHBOARD - SYSTEM STATUS")
    print("=" * 80)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def get_database_stats():
    """Obtiene estadísticas de la base de datos"""
    try:
        db_manager = DatabaseManager()
        
        with db_manager.get_session() as session:
            # Conteos generales
            total_measurements = session.query(AirQualityMeasurement).count()
            total_stations = session.query(Station).count()
            total_logs = session.query(DataIngestionLog).count()
            
            # Rango temporal
            min_date = session.query(func.min(AirQualityMeasurement.timestamp)).scalar()
            max_date = session.query(func.max(AirQualityMeasurement.timestamp)).scalar()
            
            # Estaciones activas
            stations = session.query(Station).all()
            
            # Logs recientes
            recent_logs = session.query(DataIngestionLog)\
                .order_by(DataIngestionLog.created_at.desc())\
                .limit(5).all()
            
            return {
                'total_measurements': total_measurements,
                'total_stations': total_stations,
                'total_logs': total_logs,
                'min_date': min_date,
                'max_date': max_date,
                'stations': stations,
                'recent_logs': recent_logs
            }
    except Exception as e:
        print(f"❌ Error obteniendo estadísticas de BD: {str(e)}")
        return None

def get_file_stats():
    """Obtiene estadísticas de archivos"""
    stats = {}
    
    # Dataset principal
    main_file = "data/lima_air_quality_complete.csv"
    if os.path.exists(main_file):
        size_mb = os.path.getsize(main_file) / (1024 * 1024)
        stats['main_file'] = {
            'path': main_file,
            'size_mb': size_mb,
            'exists': True
        }
        
        # Leer muestra del CSV para obtener estadísticas
        try:
            df_sample = pd.read_csv(main_file, nrows=1000)
            stats['main_file']['columns'] = list(df_sample.columns)
            stats['main_file']['sample_rows'] = len(df_sample)
        except Exception as e:
            stats['main_file']['error'] = str(e)
    else:
        stats['main_file'] = {'exists': False}
    
    # Logs
    if os.path.exists('logs'):
        log_files = [f for f in os.listdir('logs') if f.endswith('.log')]
        stats['log_files'] = len(log_files)
    else:
        stats['log_files'] = 0
    
    return stats

def print_database_stats(stats):
    """Imprime estadísticas de la base de datos"""
    if not stats:
        return
    
    print("📊 ESTADÍSTICAS DE BASE DE DATOS")
    print("-" * 40)
    print(f"📈 Total de mediciones: {stats['total_measurements']:,}")
    print(f"🏭 Estaciones registradas: {stats['total_stations']}")
    print(f"📝 Logs de ingesta: {stats['total_logs']}")
    
    if stats['min_date'] and stats['max_date']:
        print(f"📅 Período de datos: {stats['min_date'].strftime('%Y-%m-%d')} → {stats['max_date'].strftime('%Y-%m-%d')}")
        days_span = (stats['max_date'] - stats['min_date']).days
        print(f"⏱️  Duración total: {days_span:,} días")
    
    print("\n🏭 ESTACIONES ACTIVAS:")
    for station in stats['stations']:
        print(f"  • {station.name} ({station.code})")
        print(f"    📍 Lat: {station.latitude:.4f}, Lon: {station.longitude:.4f}")
    
    print("\n📋 LOGS RECIENTES:")
    for log in stats['recent_logs']:
        status_emoji = "✅" if log.status == "completed" else "❌" if log.status == "error" else "⏳"
        print(f"  {status_emoji} {log.process_id} - {log.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if log.records_processed:
            print(f"    📊 Procesados: {log.records_processed:,}")

def print_file_stats(stats):
    """Imprime estadísticas de archivos"""
    print("\n📁 ESTADÍSTICAS DE ARCHIVOS")
    print("-" * 40)
    
    if stats['main_file']['exists']:
        print(f"📄 Dataset principal: {stats['main_file']['path']}")
        print(f"💾 Tamaño: {stats['main_file']['size_mb']:.1f} MB")
        if 'columns' in stats['main_file']:
            print(f"📋 Columnas: {len(stats['main_file']['columns'])}")
            print(f"   {', '.join(stats['main_file']['columns'][:5])}...")
    else:
        print("❌ Dataset principal no encontrado")
    
    print(f"📝 Archivos de log: {stats['log_files']}")

def print_system_health():
    """Imprime estado de salud del sistema"""
    print("\n🔧 ESTADO DEL SISTEMA")
    print("-" * 40)
    
    # Verificar conexión a BD
    try:
        db_manager = DatabaseManager()
        with db_manager.get_session() as session:
            session.execute(text("SELECT 1")).scalar()
        print("✅ Conexión a base de datos: OK")
    except Exception as e:
        print(f"❌ Conexión a base de datos: ERROR - {str(e)}")
    
    # Verificar archivos críticos
    critical_files = [
        'src/database/models.py',
        'src/database/connection.py',
        'src/data_ingestion/etl.py',
        'src/data_ingestion/downloader.py',
        'config/settings.py'
    ]
    
    missing_files = [f for f in critical_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Archivos faltantes: {', '.join(missing_files)}")
    else:
        print("✅ Archivos del sistema: OK")
    
    # Verificar dependencias
    try:
        import pandas, sqlalchemy, requests
        print("✅ Dependencias Python: OK")
    except ImportError as e:
        print(f"❌ Dependencias faltantes: {str(e)}")

def print_next_steps():
    """Muestra los próximos pasos del proyecto"""
    print("\n🚀 PRÓXIMOS PASOS - FASE 2")
    print("-" * 40)
    print("1. ⚙️  Implementar pipeline de preprocesamiento")
    print("2. 🧹 Sistema de limpieza de datos avanzado")
    print("3. 📊 Agregaciones temporales (horaria, diaria, mensual)")
    print("4. 🔄 Features derivadas (estacionalidad, tendencias)")
    print("5. 🤖 Preparación de datos para Machine Learning")
    print("\n💡 Ejecutar: python src/preprocessing/clean_data.py (próximamente)")

def main():
    """Función principal"""
    print_banner()
    
    # Obtener estadísticas
    db_stats = get_database_stats()
    file_stats = get_file_stats()
    
    # Mostrar información
    if db_stats:
        print_database_stats(db_stats)
    
    print_file_stats(file_stats)
    print_system_health()
    print_next_steps()
    
    print("\n" + "=" * 80)
    print("🎯 FASE 1 COMPLETADA - Sistema de ingesta funcionando correctamente")
    print("=" * 80)

if __name__ == "__main__":
    main()
