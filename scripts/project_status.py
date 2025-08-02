#!/usr/bin/env python3
"""
Reporte completo del estado del proyecto - Todas las fases
"""

import os
import pandas as pd
from datetime import datetime

def print_banner():
    print("="*80)
    print("🌟 LIMA AIR QUALITY DASHBOARD - REPORTE COMPLETO")
    print("="*80)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_project_status():
    """Verifica el estado completo del proyecto"""
    
    print("📋 ESTADO DE LAS FASES")
    print("-" * 50)
    print("✅ Fase 1: Ingesta y Almacenamiento - COMPLETADA")
    print("✅ Fase 2: Preprocesamiento y Limpieza - COMPLETADA")
    print("🔄 Fase 3: Modelado y Predicción ML - PENDIENTE")
    print("🔄 Fase 4: API Backend - PENDIENTE")
    print("🔄 Fase 5: Dashboard Web - PENDIENTE")
    print("🔄 Fase 6: Sistema de Notificaciones - PENDIENTE")
    print("🔄 Fase 7: Despliegue y Producción - PENDIENTE")
    
    print("\n📊 DATOS PROCESADOS")
    print("-" * 50)
    
    # Verificar archivos principales
    files_info = {
        "data/lima_air_quality_complete.csv": "Dataset original consolidado",
        "data/lima_air_quality_cleaned.csv": "Datos limpios y validados",
        "data/lima_air_quality_features.csv": "Datos con características derivadas",
        "data/aggregated/lima_air_quality_daily.csv": "Agregación diaria",
        "data/aggregated/lima_air_quality_monthly.csv": "Agregación mensual",
        "data/aggregated/lima_air_quality_station_summary.csv": "Resumen por estación"
    }
    
    total_size = 0
    for file_path, description in files_info.items():
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            total_size += size_mb
            
            # Leer información básica del archivo
            try:
                df = pd.read_csv(file_path, nrows=1)
                cols = len(df.columns)
                
                # Contar todas las filas
                total_rows = sum(1 for _ in open(file_path)) - 1  # -1 para header
                
                print(f"✅ {os.path.basename(file_path)}")
                print(f"   📄 {description}")
                print(f"   📊 {total_rows:,} registros, {cols} columnas")
                print(f"   💾 {size_mb:.1f} MB")
                print()
            except:
                print(f"✅ {os.path.basename(file_path)}: {size_mb:.1f} MB (no pudo leer contenido)")
        else:
            print(f"❌ {os.path.basename(file_path)}: No encontrado")
    
    print(f"📁 TAMAÑO TOTAL DE DATOS: {total_size:.1f} MB")
    
    print("\n🔧 ESTADÍSTICAS DE PROCESAMIENTO")
    print("-" * 50)
    
    # Leer datos limpios para estadísticas
    if os.path.exists("data/lima_air_quality_features.csv"):
        df = pd.read_csv("data/lima_air_quality_features.csv")
        
        print(f"📈 Registros procesados: {len(df):,}")
        print(f"🏭 Estaciones: {df['station_name'].nunique()}")
        
        # Período temporal
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()
        days = (max_date - min_date).days
        
        print(f"📅 Período: {min_date.strftime('%Y-%m-%d')} → {max_date.strftime('%Y-%m-%d')}")
        print(f"⏱️  Duración: {days:,} días ({days/365:.1f} años)")
        
        # Contaminantes
        pollutants = ['pm10', 'no2', 'pm25', 'so2', 'o3', 'co']
        available = [col for col in pollutants if col in df.columns]
        
        print(f"\n💨 CONTAMINANTES MONITOREADOS:")
        for pol in available:
            non_null = df[pol].notna().sum()
            if non_null > 0:
                mean_val = df[pol].mean()
                print(f"  • {pol.upper()}: {non_null:,} mediciones (promedio: {mean_val:.1f} µg/m³)")
        
        # Estaciones
        print(f"\n🏭 ESTACIONES MONITOREADAS:")
        for station in sorted(df['station_name'].unique()):
            station_data = df[df['station_name'] == station]
            count = len(station_data)
            print(f"  • {station}: {count:,} registros")
    
    print("\n🛠️  INFRAESTRUCTURA TÉCNICA")
    print("-" * 50)
    
    # Verificar archivos del sistema
    system_files = [
        "src/database/models.py",
        "src/database/connection.py", 
        "src/data_ingestion/etl.py",
        "src/data_ingestion/downloader.py",
        "src/preprocessing/data_cleaner.py",
        "src/preprocessing/feature_engineer.py",
        "src/preprocessing/aggregators.py",
        "config/settings.py",
        "requirements.txt"
    ]
    
    existing_files = [f for f in system_files if os.path.exists(f)]
    print(f"✅ Archivos del sistema: {len(existing_files)}/{len(system_files)} implementados")
    
    # Verificar logs
    if os.path.exists('logs'):
        log_files = [f for f in os.listdir('logs') if f.endswith('.log')]
        print(f"📝 Archivos de log: {len(log_files)} generados")
    
    print("\n🎯 PRÓXIMOS PASOS")
    print("-" * 50)
    print("1. 🤖 Fase 3: Implementar modelos de Machine Learning")
    print("   • Modelos de predicción de PM10 y NO2")
    print("   • Análisis de series temporales")
    print("   • Clasificación de calidad del aire")
    print()
    print("2. 🔗 Fase 4: Desarrollar API Backend")
    print("   • Endpoints REST para datos")
    print("   • Sistema de autenticación")
    print("   • Documentación con Swagger")
    print()
    print("3. 🌐 Fase 5: Crear Dashboard Web")
    print("   • Visualizaciones interactivas")
    print("   • Mapas de calidad del aire")
    print("   • Alertas en tiempo real")

def main():
    print_banner()
    check_project_status()
    
    print("\n" + "="*80)
    print("🎊 RESUMEN: 2 DE 7 FASES COMPLETADAS (28.6%)")
    print("🚀 SISTEMA LISTO PARA FASE 3: MODELADO ML")
    print("="*80)

if __name__ == "__main__":
    main()
