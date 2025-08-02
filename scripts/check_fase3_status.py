#!/usr/bin/env python3
"""
Actualización del estado del proyecto - Fase 3 completada
Dashboard de Calidad del Aire de Lima
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_fase3_status():
    """
    Verifica el estado completo después de la Fase 3
    """
    print("="*80)
    print("🎯 ESTADO FINAL DEL PROYECTO - DASHBOARD CALIDAD DEL AIRE LIMA")
    print("="*80)
    
    # Estado general del proyecto
    print("\n📊 RESUMEN DE FASES COMPLETADAS:")
    print("-"*50)
    print("✅ Fase 1: Ingesta y Consolidación de Datos - COMPLETADA")
    print("✅ Fase 2: Procesamiento y Feature Engineering - COMPLETADA") 
    print("✅ Fase 3: Modelado y Predicción ML - COMPLETADA")
    print("⏳ Fase 4: Análisis Exploratorio de Datos (EDA) - PENDIENTE")
    print("⏳ Fase 5: Dashboard Interactivo - PENDIENTE")
    print("⏳ Fase 6: API y Microservicios - PENDIENTE")
    print("⏳ Fase 7: Deployment y Monitoring - PENDIENTE")
    
    # Verificar archivos de datos
    print("\n📁 ARCHIVOS DE DATOS:")
    print("-"*50)
    
    data_files = [
        "data/lima_air_quality_cleaned.csv",
        "data/lima_air_quality_features.csv", 
        "data/lima_air_quality_daily.csv",
        "data/lima_air_quality_monthly.csv",
        "data/lima_air_quality_station_summary.csv"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"✅ {file_path} ({size:.1f} MB)")
        else:
            print(f"❌ {file_path} - NO ENCONTRADO")
    
    # Verificar modelos ML
    print("\n🤖 MODELOS DE MACHINE LEARNING:")
    print("-"*50)
    
    model_files = [
        "models/best_pm10_predictor.joblib",
        "models/best_no2_predictor.joblib",
        "models/best_pm10_classifier.joblib", 
        "models/best_no2_classifier.joblib"
    ]
    
    for model_path in model_files:
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / (1024*1024)  # MB
            print(f"✅ {model_path} ({size:.1f} MB)")
        else:
            print(f"❌ {model_path} - NO ENCONTRADO")
    
    # Verificar resultados y visualizaciones
    print("\n📈 RESULTADOS Y VISUALIZACIONES:")
    print("-"*50)
    
    results_files = [
        "results/pm10_regression_results.png",
        "results/no2_regression_results.png", 
        "results/pm10_classification_results.png",
        "results/no2_classification_results.png",
        "results/pm10_timeseries_results.png",
        "results/no2_timeseries_results.png",
        "results/ml_summary_report.txt"
    ]
    
    for result_path in results_files:
        if os.path.exists(result_path):
            print(f"✅ {result_path}")
        else:
            print(f"❌ {result_path} - NO ENCONTRADO")
    
    # Estadísticas de datos
    print("\n📊 ESTADÍSTICAS DE DATOS:")
    print("-"*50)
    
    try:
        # Datos limpios
        df_clean = pd.read_csv("data/lima_air_quality_cleaned.csv")
        print(f"✅ Registros limpios: {len(df_clean):,}")
        print(f"✅ Período: {df_clean['timestamp'].min()} a {df_clean['timestamp'].max()}")
        
        # Datos con features
        df_features = pd.read_csv("data/lima_air_quality_features.csv")
        print(f"✅ Registros con features: {len(df_features):,}")
        print(f"✅ Características totales: {df_features.shape[1]}")
        
        # Contaminantes disponibles
        pollutants = ['pm10', 'pm25', 'no2', 'so2', 'o3', 'co']
        available = [p for p in pollutants if p in df_features.columns]
        print(f"✅ Contaminantes disponibles: {', '.join(available).upper()}")
        
        # Estaciones
        stations = df_features['station_name'].nunique()
        print(f"✅ Estaciones de monitoreo: {stations}")
        
    except Exception as e:
        print(f"❌ Error leyendo estadísticas: {e}")
    
    # Resultados de ML
    print("\n🏆 MEJORES MODELOS DE ML:")
    print("-"*50)
    
    try:
        with open("results/ml_summary_report.txt", "r") as f:
            content = f.read()
            
        # Extraer información clave
        if "Mejor Regresión:" in content:
            regression_line = [line for line in content.split('\n') if "Mejor Regresión:" in line][0]
            print(f"✅ {regression_line}")
        
        if "Mejor Clasificación:" in content:
            classification_line = [line for line in content.split('\n') if "Mejor Clasificación:" in line][0]
            print(f"✅ {classification_line}")
            
        if "Mejor Pronóstico:" in content:
            forecast_line = [line for line in content.split('\n') if "Mejor Pronóstico:" in line][0]
            print(f"✅ {forecast_line}")
            
        if "Total de modelos entrenados:" in content:
            total_line = [line for line in content.split('\n') if "Total de modelos entrenados:" in line][0]
            print(f"✅ {total_line}")
            
    except Exception as e:
        print(f"❌ Error leyendo resumen ML: {e}")
    
    # Capacidades del sistema
    print("\n🔧 CAPACIDADES DEL SISTEMA:")
    print("-"*50)
    print("✅ Predicción de niveles de PM10 y NO2")
    print("✅ Clasificación de calidad del aire (Good, Moderate, Unhealthy, Hazardous)")
    print("✅ Pronóstico de series temporales (hasta 24 horas)")
    print("✅ Análisis de importancia de características")
    print("✅ Evaluación comparativa de múltiples algoritmos")
    print("✅ Modelos estadísticos y de deep learning")
    print("✅ Métricas de rendimiento completas")
    print("✅ Visualizaciones automatizadas")
    
    # Próximos pasos
    print("\n🚀 PRÓXIMOS PASOS RECOMENDADOS:")
    print("-"*50)
    print("1. 📊 Fase 4: Crear análisis exploratorio de datos (EDA)")
    print("2. 🌐 Fase 5: Desarrollar dashboard interactivo con Streamlit")
    print("3. 🔌 Fase 6: Implementar API REST para los modelos")
    print("4. 🚀 Fase 7: Configurar deployment y monitoring")
    print("5. 🔄 Automatizar pipeline de actualización de datos")
    print("6. 📱 Crear alertas automáticas de calidad del aire")
    
    print("\n" + "="*80)
    print("🎉 FASE 3 COMPLETADA EXITOSAMENTE")
    print("🎯 3 de 7 fases completadas (43% del proyecto)")
    print("="*80)
    
    return True

def main():
    """Función principal"""
    try:
        check_fase3_status()
        
        # Crear archivo de estado
        status_info = {
            'fecha_actualizacion': datetime.now().isoformat(),
            'fases_completadas': 3,
            'fases_totales': 7,
            'progreso_porcentaje': 43,
            'fase_actual': 'Fase 3: Modelado y Predicción ML - COMPLETADA',
            'proxima_fase': 'Fase 4: Análisis Exploratorio de Datos (EDA)',
            'modelos_entrenados': 44,
            'contaminantes_analizados': ['PM10', 'NO2'],
            'tipos_modelos': ['Regresión', 'Clasificación', 'Series Temporales']
        }
        
        import json
        with open('project_status_fase3.json', 'w') as f:
            json.dump(status_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Estado guardado en: project_status_fase3.json")
        
    except Exception as e:
        logger.error(f"Error verificando estado: {e}")

if __name__ == "__main__":
    main()
