#!/usr/bin/env python3
"""
Pipeline principal de Machine Learning - Fase 3
Ejecuta todos los modelos de ML y genera resultados completos
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Agregar el directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fase3_ml_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Pipeline principal de Machine Learning
    """
    try:
        logger.info("🚀 INICIANDO FASE 3: MODELADO Y PREDICCIÓN ML")
        logger.info("=" * 60)
        
        # Crear directorios necesarios
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Importar módulos ML
        try:
            # Agregar ruta específica para imports
            ml_path = os.path.join(os.path.dirname(__file__), 'src', 'ml')
            if ml_path not in sys.path:
                sys.path.insert(0, ml_path)
            
            from data_preparation import DataPreparator
            from pollution_predictor import PollutionPredictor
            from timeseries_forecaster import TimeSeriesForecaster
            from air_quality_classifier import AirQualityClassifier
        except ImportError as e:
            logger.error(f"Error importando módulos ML: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 1. CARGAR DATOS
        logger.info("📊 Paso 1: Cargando datos procesados...")
        
        if not os.path.exists("data/lima_air_quality_features.csv"):
            logger.error("Archivo con características no encontrado. Ejecutar Fase 2 primero.")
            return
        
        df = pd.read_csv("data/lima_air_quality_features.csv")
        logger.info(f"   ✅ Datos cargados: {len(df):,} registros, {df.shape[1]} columnas")
        
        # Verificar contaminantes disponibles
        pollutants = ['pm10', 'pm25', 'no2', 'so2', 'o3', 'co']
        available_pollutants = [p for p in pollutants if p in df.columns]
        logger.info(f"   ✅ Contaminantes disponibles: {available_pollutants}")
        
        if not available_pollutants:
            logger.error("No se encontraron contaminantes en los datos")
            return
        
        # 2. PREPARACIÓN DE DATOS
        logger.info("\n🔧 Paso 2: Preparando datos para ML...")
        
        preparator = DataPreparator()
        prepared_data = {}
        
        # Preparar datos para cada contaminante principal
        main_pollutants = ['pm10', 'no2']  # Empezar con los principales
        main_pollutants = [p for p in main_pollutants if p in available_pollutants]
        
        for pollutant in main_pollutants:
            logger.info(f"   Preparando datos para {pollutant.upper()}...")
            
            # Datos para regresión
            try:
                regression_data = preparator.prepare_regression_data(df, pollutant)
                prepared_data[f'{pollutant}_regression'] = regression_data
                logger.info(f"     ✅ Regresión {pollutant}: {regression_data['X_train'].shape}")
            except Exception as e:
                logger.error(f"     ❌ Error en regresión {pollutant}: {e}")
            
            # Datos para clasificación
            try:
                classification_data = preparator.prepare_classification_data(df, pollutant)
                prepared_data[f'{pollutant}_classification'] = classification_data
                logger.info(f"     ✅ Clasificación {pollutant}: {classification_data['X_train'].shape}")
            except Exception as e:
                logger.error(f"     ❌ Error en clasificación {pollutant}: {e}")
            
            # Datos para series temporales
            try:
                timeseries_data = preparator.prepare_timeseries_data(df, pollutant, sequence_length=24)
                prepared_data[f'{pollutant}_timeseries'] = timeseries_data
                logger.info(f"     ✅ Series temporales {pollutant}: {timeseries_data['X_train'].shape}")
            except Exception as e:
                logger.error(f"     ❌ Error en series temporales {pollutant}: {e}")
        
        # 3. MODELOS DE REGRESIÓN
        logger.info("\n🤖 Paso 3: Entrenando modelos de regresión...")
        
        regression_results = {}
        for pollutant in main_pollutants:
            data_key = f'{pollutant}_regression'
            if data_key in prepared_data:
                logger.info(f"   Entrenando predictor de {pollutant.upper()}...")
                
                predictor = PollutionPredictor()
                data = prepared_data[data_key]
                
                try:
                    results = predictor.train_multiple_models(
                        data['X_train'], data['y_train'],
                        data['X_test'], data['y_test'],
                        data['feature_names']
                    )
                    
                    # Crear visualizaciones
                    predictor.plot_results(
                        results, data['y_test'],
                        save_path=f"results/{pollutant}_regression_results.png"
                    )
                    
                    predictor.plot_feature_importance_comparison(
                        results,
                        save_path=f"results/{pollutant}_feature_importance.png"
                    )
                    
                    # Guardar modelo
                    predictor.save_best_model(f"models/best_{pollutant}_predictor.joblib")
                    
                    regression_results[pollutant] = {
                        'predictor': predictor,
                        'results': results,
                        'best_model': predictor.best_model_name,
                        'best_r2': max([r['metrics']['test_r2'] for r in results.values()])
                    }
                    
                    logger.info(f"     ✅ Mejor modelo {pollutant}: {predictor.best_model_name} "
                               f"(R² = {regression_results[pollutant]['best_r2']:.4f})")
                    
                except Exception as e:
                    logger.error(f"     ❌ Error entrenando predictor {pollutant}: {e}")
        
        # 4. MODELOS DE CLASIFICACIÓN
        logger.info("\n🎯 Paso 4: Entrenando modelos de clasificación...")
        
        classification_results = {}
        for pollutant in main_pollutants:
            data_key = f'{pollutant}_classification'
            if data_key in prepared_data:
                logger.info(f"   Entrenando clasificador de calidad {pollutant.upper()}...")
                
                classifier = AirQualityClassifier()
                data = prepared_data[data_key]
                
                try:
                    results = classifier.train_multiple_classifiers(
                        data['X_train'], data['y_train'],
                        data['X_test'], data['y_test'],
                        data['feature_names'], data['class_names']
                    )
                    
                    # Crear visualizaciones
                    classifier.plot_classification_results(
                        results, data['y_test'],
                        save_path=f"results/{pollutant}_classification_results.png"
                    )
                    
                    classifier.plot_roc_curves(
                        results, data['y_test'],
                        save_path=f"results/{pollutant}_roc_curves.png"
                    )
                    
                    # Generar y guardar reporte
                    report = classifier.generate_classification_report(results, data['y_test'])
                    with open(f"results/{pollutant}_classification_report.txt", "w") as f:
                        f.write(report)
                    
                    # Guardar modelo
                    classifier.save_best_model(f"models/best_{pollutant}_classifier.joblib")
                    
                    classification_results[pollutant] = {
                        'classifier': classifier,
                        'results': results,
                        'best_model': classifier.best_model_name,
                        'best_accuracy': max([r['metrics']['test_accuracy'] for r in results.values()])
                    }
                    
                    logger.info(f"     ✅ Mejor clasificador {pollutant}: {classifier.best_model_name} "
                               f"(Accuracy = {classification_results[pollutant]['best_accuracy']:.4f})")
                    
                except Exception as e:
                    logger.error(f"     ❌ Error entrenando clasificador {pollutant}: {e}")
        
        # 5. MODELOS DE SERIES TEMPORALES
        logger.info("\n📈 Paso 5: Entrenando modelos de series temporales...")
        
        timeseries_results = {}
        for pollutant in main_pollutants:
            data_key = f'{pollutant}_timeseries'
            if data_key in prepared_data:
                logger.info(f"   Entrenando forecaster de {pollutant.upper()}...")
                
                forecaster = TimeSeriesForecaster()
                data = prepared_data[data_key]
                
                try:
                    results = forecaster.train_all_models(
                        data['X_train'], data['y_train'],
                        data['X_test'], data['y_test'],
                        data['sequence_length']
                    )
                    
                    # Crear visualizaciones
                    forecaster.plot_timeseries_results(
                        results, data['y_test'],
                        save_path=f"results/{pollutant}_timeseries_results.png"
                    )
                    
                    # Historial de entrenamiento para modelos de deep learning
                    try:
                        forecaster.plot_training_history(
                            results,
                            save_path=f"results/{pollutant}_training_history.png"
                        )
                    except:
                        pass  # No hay modelos de deep learning
                    
                    # Guardar modelo
                    forecaster.save_best_model(f"models/best_{pollutant}_forecaster")
                    
                    timeseries_results[pollutant] = {
                        'forecaster': forecaster,
                        'results': results,
                        'best_model': forecaster.best_model_name,
                        'best_r2': max([r['metrics']['test_r2'] for r in results.values()])
                    }
                    
                    logger.info(f"     ✅ Mejor forecaster {pollutant}: {forecaster.best_model_name} "
                               f"(R² = {timeseries_results[pollutant]['best_r2']:.4f})")
                    
                except Exception as e:
                    logger.error(f"     ❌ Error entrenando forecaster {pollutant}: {e}")
        
        # 6. RESUMEN DE RESULTADOS
        logger.info("\n📋 Paso 6: Generando resumen de resultados...")
        
        summary = generate_ml_summary(regression_results, classification_results, timeseries_results)
        
        # Guardar resumen
        with open("results/ml_summary_report.txt", "w") as f:
            f.write(summary)
        
        print(summary)
        
        logger.info("\n🎉 FASE 3 COMPLETADA EXITOSAMENTE")
        logger.info("=" * 60)
        logger.info("📁 Resultados guardados en:")
        logger.info("   - models/: Modelos entrenados")
        logger.info("   - results/: Gráficas y reportes")
        logger.info("   - logs/: Logs de ejecución")
        
    except Exception as e:
        logger.error(f"❌ Error en el pipeline de ML: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_ml_summary(regression_results, classification_results, timeseries_results):
    """
    Genera un resumen completo de todos los resultados de ML
    """
    summary = f"\n{'='*80}\n"
    summary += f"RESUMEN COMPLETO - FASE 3: MODELADO Y PREDICCIÓN ML\n"
    summary += f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary += f"{'='*80}\n\n"
    
    # Resumen de regresión
    summary += "🤖 MODELOS DE REGRESIÓN (Predicción de Contaminantes)\n"
    summary += "-" * 60 + "\n"
    
    for pollutant, result in regression_results.items():
        summary += f"\n{pollutant.upper()}:\n"
        summary += f"  Mejor Modelo: {result['best_model']}\n"
        summary += f"  R² Score: {result['best_r2']:.4f}\n"
        
        # Top 3 modelos
        sorted_models = sorted(result['results'].items(), 
                             key=lambda x: x[1]['metrics']['test_r2'], reverse=True)
        summary += f"  Top 3 Modelos:\n"
        for i, (model_name, model_result) in enumerate(sorted_models[:3]):
            summary += f"    {i+1}. {model_name}: R² = {model_result['metrics']['test_r2']:.4f}\n"
    
    # Resumen de clasificación
    summary += f"\n🎯 MODELOS DE CLASIFICACIÓN (Calidad del Aire)\n"
    summary += "-" * 60 + "\n"
    
    for pollutant, result in classification_results.items():
        summary += f"\n{pollutant.upper()}:\n"
        summary += f"  Mejor Modelo: {result['best_model']}\n"
        summary += f"  Accuracy: {result['best_accuracy']:.4f}\n"
        
        # Top 3 modelos
        sorted_models = sorted(result['results'].items(), 
                             key=lambda x: x[1]['metrics']['test_accuracy'], reverse=True)
        summary += f"  Top 3 Modelos:\n"
        for i, (model_name, model_result) in enumerate(sorted_models[:3]):
            summary += f"    {i+1}. {model_name}: Accuracy = {model_result['metrics']['test_accuracy']:.4f}\n"
    
    # Resumen de series temporales
    summary += f"\n📈 MODELOS DE SERIES TEMPORALES (Pronóstico)\n"
    summary += "-" * 60 + "\n"
    
    for pollutant, result in timeseries_results.items():
        summary += f"\n{pollutant.upper()}:\n"
        summary += f"  Mejor Modelo: {result['best_model']}\n"
        summary += f"  R² Score: {result['best_r2']:.4f}\n"
        
        # Top 3 modelos
        sorted_models = sorted(result['results'].items(), 
                             key=lambda x: x[1]['metrics']['test_r2'], reverse=True)
        summary += f"  Top 3 Modelos:\n"
        for i, (model_name, model_result) in enumerate(sorted_models[:3]):
            summary += f"    {i+1}. {model_name}: R² = {model_result['metrics']['test_r2']:.4f}\n"
    
    # Estadísticas generales
    summary += f"\n📊 ESTADÍSTICAS GENERALES\n"
    summary += "-" * 60 + "\n"
    
    total_models = sum(len(r['results']) for r in regression_results.values())
    total_models += sum(len(r['results']) for r in classification_results.values())
    total_models += sum(len(r['results']) for r in timeseries_results.values())
    
    summary += f"Total de modelos entrenados: {total_models}\n"
    summary += f"Contaminantes analizados: {len(set(list(regression_results.keys()) + list(classification_results.keys()) + list(timeseries_results.keys())))}\n"
    summary += f"Tipos de modelos: Regresión, Clasificación, Series Temporales\n"
    
    # Mejores modelos generales
    summary += f"\n🏆 MEJORES MODELOS POR CATEGORÍA\n"
    summary += "-" * 60 + "\n"
    
    if regression_results:
        best_regression = max(regression_results.items(), 
                            key=lambda x: x[1]['best_r2'])
        summary += f"Mejor Regresión: {best_regression[0].upper()} - {best_regression[1]['best_model']} (R² = {best_regression[1]['best_r2']:.4f})\n"
    
    if classification_results:
        best_classification = max(classification_results.items(), 
                                key=lambda x: x[1]['best_accuracy'])
        summary += f"Mejor Clasificación: {best_classification[0].upper()} - {best_classification[1]['best_model']} (Acc = {best_classification[1]['best_accuracy']:.4f})\n"
    
    if timeseries_results:
        best_timeseries = max(timeseries_results.items(), 
                            key=lambda x: x[1]['best_r2'])
        summary += f"Mejor Pronóstico: {best_timeseries[0].upper()} - {best_timeseries[1]['best_model']} (R² = {best_timeseries[1]['best_r2']:.4f})\n"
    
    summary += f"\n{'='*80}\n"
    summary += "Todos los modelos y resultados han sido guardados en las carpetas 'models/' y 'results/'\n"
    summary += f"{'='*80}\n"
    
    return summary

if __name__ == "__main__":
    main()
