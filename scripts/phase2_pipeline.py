#!/usr/bin/env python3
"""
Pipeline principal de Fase 2: Preprocesamiento y limpieza de datos
Dashboard de Calidad del Aire de Lima
"""

import os
import sys
import time
import logging
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.feature_engineer import FeatureEngineer
from src.preprocessing.aggregators import TimeSeriesAggregator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/preprocessing_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """
    Pipeline completo de preprocesamiento de datos
    """
    
    def __init__(self):
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.aggregator = TimeSeriesAggregator()
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'duration': None,
            'stages_completed': [],
            'total_records_processed': 0,
            'final_datasets_created': 0
        }
    
    def run_full_pipeline(self, input_file: str = None) -> dict:
        """
        Ejecuta el pipeline completo de preprocesamiento
        
        Args:
            input_file: Archivo de entrada (opcional)
            
        Returns:
            Diccionario con resultados del pipeline
        """
        self.pipeline_stats['start_time'] = datetime.now()
        logger.info("🚀 INICIANDO PIPELINE DE PREPROCESAMIENTO - FASE 2")
        logger.info("="*80)
        
        try:
            # Etapa 1: Limpieza de datos
            logger.info("📊 ETAPA 1: LIMPIEZA DE DATOS")
            logger.info("-" * 40)
            
            if input_file is None:
                input_file = "data/lima_air_quality_complete.csv"
            
            # Verificar que el archivo existe
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Archivo de entrada no encontrado: {input_file}")
            
            cleaned_df = self.cleaner.clean_dataset(
                input_file=input_file,
                output_file="data/lima_air_quality_cleaned.csv"
            )
            
            self.pipeline_stats['stages_completed'].append('cleaning')
            self.pipeline_stats['total_records_processed'] = len(cleaned_df)
            
            logger.info(f"✅ Etapa 1 completada: {len(cleaned_df):,} registros limpios")
            
            # Etapa 2: Ingeniería de características
            logger.info("\n🔧 ETAPA 2: INGENIERÍA DE CARACTERÍSTICAS")
            logger.info("-" * 40)
            
            featured_df = self.engineer.create_all_features(cleaned_df)
            featured_df.to_csv("data/lima_air_quality_features.csv", index=False)
            
            self.pipeline_stats['stages_completed'].append('feature_engineering')
            
            logger.info(f"✅ Etapa 2 completada: {len(featured_df.columns)} características totales")
            
            # Etapa 3: Agregaciones temporales
            logger.info("\n📈 ETAPA 3: AGREGACIONES TEMPORALES")
            logger.info("-" * 40)
            
            aggregations = self.aggregator.create_all_aggregations(featured_df)
            self.aggregator.save_aggregations("data/aggregated")
            
            self.pipeline_stats['stages_completed'].append('aggregation')
            self.pipeline_stats['final_datasets_created'] = len(aggregations)
            
            logger.info(f"✅ Etapa 3 completada: {len(aggregations)} datasets agregados")
            
            # Completar pipeline
            self.pipeline_stats['end_time'] = datetime.now()
            self.pipeline_stats['duration'] = (
                self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']
            ).total_seconds()
            
            # Generar reporte final
            self._generate_final_report(aggregations)
            
            return {
                'success': True,
                'stats': self.pipeline_stats,
                'cleaned_records': len(cleaned_df),
                'featured_records': len(featured_df),
                'aggregated_datasets': len(aggregations),
                'output_files': self._get_output_files()
            }
            
        except Exception as e:
            logger.error(f"❌ Error en pipeline: {str(e)}")
            self.pipeline_stats['end_time'] = datetime.now()
            self.pipeline_stats['duration'] = (
                self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']
            ).total_seconds()
            
            return {
                'success': False,
                'error': str(e),
                'stats': self.pipeline_stats
            }
    
    def _generate_final_report(self, aggregations: dict):
        """Genera reporte final del pipeline"""
        logger.info("\n" + "="*80)
        logger.info("🎯 REPORTE FINAL - FASE 2 COMPLETADA")
        logger.info("="*80)
        
        # Estadísticas del pipeline
        logger.info(f"⏱️  Duración total: {self.pipeline_stats['duration']:.2f} segundos")
        logger.info(f"📊 Registros procesados: {self.pipeline_stats['total_records_processed']:,}")
        logger.info(f"🔧 Etapas completadas: {len(self.pipeline_stats['stages_completed'])}/3")
        logger.info(f"📁 Datasets generados: {self.pipeline_stats['final_datasets_created']}")
        
        # Estadísticas de limpieza
        if hasattr(self.cleaner, 'stats'):
            stats = self.cleaner.stats
            logger.info(f"\n📋 ESTADÍSTICAS DE LIMPIEZA:")
            logger.info(f"  • Registros originales: {stats['total_records']:,}")
            logger.info(f"  • Registros finales: {stats['cleaned_records']:,}")
            logger.info(f"  • Duplicados removidos: {stats['removed_duplicates']:,}")
            logger.info(f"  • Outliers removidos: {stats['removed_outliers']:,}")
            logger.info(f"  • Valores imputados: {stats['imputed_values']:,}")
            
            retention_rate = (stats['cleaned_records'] / stats['total_records']) * 100
            logger.info(f"  • Tasa de retención: {retention_rate:.1f}%")
        
        # Características creadas
        if hasattr(self.engineer, 'features_created'):
            logger.info(f"\n🔧 CARACTERÍSTICAS CREADAS: {len(self.engineer.features_created)}")
            categories = {
                'temporal': [f for f in self.engineer.features_created if any(x in f for x in ['hour', 'day', 'month', 'year', 'season'])],
                'indices': [f for f in self.engineer.features_created if 'aqi' in f or 'index' in f],
                'ratios': [f for f in self.engineer.features_created if 'ratio' in f],
                'lag': [f for f in self.engineer.features_created if 'lag' in f or 'rolling' in f],
                'statistical': [f for f in self.engineer.features_created if any(x in f for x in ['std', 'max', 'min', 'mean'])]
            }
            
            for category, features in categories.items():
                if features:
                    logger.info(f"  • {category.title()}: {len(features)} características")
        
        # Agregaciones creadas
        if aggregations:
            logger.info(f"\n📈 AGREGACIONES TEMPORALES:")
            summary = self.aggregator.get_aggregation_summary()
            for name, info in summary.items():
                logger.info(f"  • {name.title()}: {info['records']:,} registros, {info['stations']} estaciones")
        
        # Archivos de salida
        output_files = self._get_output_files()
        logger.info(f"\n📁 ARCHIVOS GENERADOS:")
        for file_path, info in output_files.items():
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"  • {file_path}: {size_mb:.1f} MB")
        
        logger.info("\n🚀 PRÓXIMO PASO: Fase 3 - Modelado y Predicción ML")
        logger.info("="*80)
    
    def _get_output_files(self) -> dict:
        """Retorna diccionario de archivos de salida"""
        return {
            "data/lima_air_quality_cleaned.csv": "Dataset limpio y validado",
            "data/lima_air_quality_features.csv": "Dataset con características derivadas",
            "data/aggregated/lima_air_quality_daily.csv": "Agregación diaria",
            "data/aggregated/lima_air_quality_weekly.csv": "Agregación semanal",
            "data/aggregated/lima_air_quality_monthly.csv": "Agregación mensual",
            "data/aggregated/lima_air_quality_station_summary.csv": "Resumen por estación"
        }
    
    def validate_output_files(self) -> dict:
        """Valida que todos los archivos de salida existan"""
        output_files = self._get_output_files()
        validation_results = {}
        
        for file_path, description in output_files.items():
            exists = os.path.exists(file_path)
            validation_results[file_path] = {
                'exists': exists,
                'description': description,
                'size_mb': os.path.getsize(file_path) / (1024 * 1024) if exists else 0
            }
        
        return validation_results

def main():
    """Función principal"""
    pipeline = PreprocessingPipeline()
    
    logger.info("Iniciando Pipeline de Preprocesamiento - Fase 2")
    
    # Ejecutar pipeline completo
    results = pipeline.run_full_pipeline()
    
    if results['success']:
        logger.info("\n🎉 ¡PIPELINE COMPLETADO EXITOSAMENTE!")
        
        # Validar archivos de salida
        validation = pipeline.validate_output_files()
        
        missing_files = [f for f, info in validation.items() if not info['exists']]
        if missing_files:
            logger.warning(f"⚠️  Archivos faltantes: {len(missing_files)}")
            for file in missing_files:
                logger.warning(f"  • {file}")
        else:
            logger.info("✅ Todos los archivos de salida generados correctamente")
        
    else:
        logger.error("❌ Pipeline falló")
        logger.error(f"Error: {results.get('error', 'Desconocido')}")

if __name__ == "__main__":
    main()
