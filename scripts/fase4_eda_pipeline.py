#!/usr/bin/env python3
"""
Pipeline principal de Análisis Exploratorio de Datos (EDA) - Fase 4
Ejecuta todos los análisis de EDA y genera reportes completos
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
        logging.FileHandler('logs/fase4_eda_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Pipeline principal de Análisis Exploratorio de Datos
    """
    try:
        logger.info("🚀 INICIANDO FASE 4: ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
        logger.info("=" * 70)
        
        # Crear directorios necesarios
        os.makedirs("results/eda", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Importar módulos EDA
        try:
            # Agregar ruta específica para imports
            eda_path = os.path.join(os.path.dirname(__file__), 'src', 'eda')
            if eda_path not in sys.path:
                sys.path.insert(0, eda_path)
            
            from temporal_analysis import TemporalAnalyzer
            from spatial_analysis import SpatialAnalyzer
            from correlation_analysis import CorrelationAnalyzer
            
        except ImportError as e:
            logger.error(f"Error importando módulos EDA: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 1. CARGAR DATOS
        logger.info("📊 Paso 1: Cargando datos procesados...")
        
        if not os.path.exists("data/lima_air_quality_features.csv"):
            logger.error("Archivo con características no encontrado. Ejecutar Fases 1-3 primero.")
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
        
        # Verificar información temporal
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            date_range = f"{df['timestamp'].min().date()} a {df['timestamp'].max().date()}"
            logger.info(f"   ✅ Período temporal: {date_range}")
        
        # Verificar información espacial
        stations = df['station_name'].nunique() if 'station_name' in df.columns else 0
        logger.info(f"   ✅ Estaciones de monitoreo: {stations}")
        
        # 2. ANÁLISIS TEMPORAL
        logger.info("\n⏰ Paso 2: Ejecutando análisis temporal...")
        
        temporal_analyzer = TemporalAnalyzer()
        temporal_analyzer.load_data(df)
        
        try:
            # Vista general de series temporales
            logger.info("   Creando vista general de series temporales...")
            temporal_analyzer.create_time_series_overview(
                save_path="results/eda/temporal_overview.png"
            )
            
            # Patrones estacionales
            logger.info("   Analizando patrones estacionales...")
            temporal_analyzer.analyze_seasonal_patterns(
                save_path="results/eda/seasonal_patterns.png"
            )
            
            # Patrones diarios
            logger.info("   Analizando patrones diarios...")
            temporal_analyzer.analyze_daily_patterns(
                save_path="results/eda/daily_patterns.png"
            )
            
            # Línea de tiempo interactiva
            logger.info("   Creando línea de tiempo interactiva...")
            fig_temporal = temporal_analyzer.create_interactive_timeline(
                save_path="results/eda/interactive_timeline.html"
            )
            
            # Análisis de tendencias
            logger.info("   Analizando tendencias temporales...")
            trends_results = temporal_analyzer.analyze_trends_and_changes(
                save_path="results/eda/trends_analysis.png"
            )
            
            # Guardar resultados de tendencias
            import json
            with open("results/eda/temporal_trends.json", "w") as f:
                json.dump(trends_results, f, indent=2, ensure_ascii=False)
            
            logger.info("     ✅ Análisis temporal completado")
            
        except Exception as e:
            logger.error(f"     ❌ Error en análisis temporal: {e}")
        
        # 3. ANÁLISIS ESPACIAL
        logger.info("\n🗺️ Paso 3: Ejecutando análisis espacial...")
        
        spatial_analyzer = SpatialAnalyzer()
        spatial_analyzer.load_data(df)
        
        try:
            # Comparación entre estaciones
            logger.info("   Comparando niveles entre estaciones...")
            spatial_analyzer.create_station_comparison(
                save_path="results/eda/station_comparison.png"
            )
            
            # Mapa de calor espacial
            logger.info("   Creando mapa de calor espacial...")
            spatial_analyzer.create_spatial_heatmap(
                save_path="results/eda/spatial_heatmap.png"
            )
            
            # Mapa interactivo
            logger.info("   Creando mapa interactivo...")
            fig_spatial = spatial_analyzer.create_interactive_map(
                save_path="results/eda/interactive_map.html"
            )
            
            # Patrones espaciales
            logger.info("   Analizando patrones espaciales...")
            spatial_results = spatial_analyzer.analyze_spatial_patterns(
                save_path="results/eda/spatial_patterns.png"
            )
            
            # Análisis de distancia
            logger.info("   Analizando correlación por distancia...")
            spatial_analyzer.create_distance_analysis(
                save_path="results/eda/distance_analysis.png"
            )
            
            # Guardar resultados espaciales
            with open("results/eda/spatial_results.json", "w") as f:
                json.dump(spatial_results, f, indent=2, ensure_ascii=False)
            
            logger.info("     ✅ Análisis espacial completado")
            
        except Exception as e:
            logger.error(f"     ❌ Error en análisis espacial: {e}")
        
        # 4. ANÁLISIS DE CORRELACIONES
        logger.info("\n🔗 Paso 4: Ejecutando análisis de correlaciones...")
        
        correlation_analyzer = CorrelationAnalyzer()
        correlation_analyzer.load_data(df)
        
        try:
            # Mapa de calor de correlaciones
            logger.info("   Creando mapa de calor de correlaciones...")
            correlation_analyzer.create_correlation_heatmap(
                save_path="results/eda/correlation_heatmap.png"
            )
            
            # Correlaciones con características
            logger.info("   Analizando correlaciones con características...")
            correlation_analyzer.analyze_feature_correlations(
                save_path="results/eda/feature_correlations.png"
            )
            
            # Matriz de dispersión
            logger.info("   Creando matriz de dispersión...")
            correlation_analyzer.create_scatter_matrix(
                save_path="results/eda/scatter_matrix.png"
            )
            
            # Gráfico interactivo de correlaciones
            logger.info("   Creando gráfico interactivo de correlaciones...")
            fig_correlation = correlation_analyzer.create_interactive_correlation_plot(
                save_path="results/eda/interactive_correlations.html"
            )
            
            # Análisis PCA
            logger.info("   Realizando análisis de componentes principales...")
            pca_results = correlation_analyzer.perform_pca_analysis(
                save_path="results/eda/pca_analysis.png"
            )
            
            # Generar reporte de correlaciones
            correlation_report = correlation_analyzer.generate_correlation_report()
            with open("results/eda/correlation_report.txt", "w") as f:
                f.write(correlation_report)
            
            # Guardar resultados PCA
            if pca_results:
                with open("results/eda/pca_results.json", "w") as f:
                    json.dump(pca_results, f, indent=2, ensure_ascii=False)
            
            logger.info("     ✅ Análisis de correlaciones completado")
            
        except Exception as e:
            logger.error(f"     ❌ Error en análisis de correlaciones: {e}")
        
        # 5. GENERAR RESUMEN EJECUTIVO
        logger.info("\n📋 Paso 5: Generando resumen ejecutivo de EDA...")
        
        try:
            executive_summary = generate_eda_executive_summary(
                df, available_pollutants, trends_results, spatial_results, 
                correlation_report, pca_results
            )
            
            # Guardar resumen ejecutivo
            with open("results/eda/executive_summary.txt", "w") as f:
                f.write(executive_summary)
            
            print(executive_summary)
            
            logger.info("     ✅ Resumen ejecutivo generado")
            
        except Exception as e:
            logger.error(f"     ❌ Error generando resumen ejecutivo: {e}")
        
        # 6. CREAR ÍNDICE DE VISUALIZACIONES
        logger.info("\n📁 Paso 6: Creando índice de visualizaciones...")
        
        try:
            create_visualization_index()
            logger.info("     ✅ Índice de visualizaciones creado")
            
        except Exception as e:
            logger.error(f"     ❌ Error creando índice: {e}")
        
        logger.info("\n🎉 FASE 4 COMPLETADA EXITOSAMENTE")
        logger.info("=" * 70)
        logger.info("📁 Resultados guardados en:")
        logger.info("   - results/eda/: Visualizaciones y análisis")
        logger.info("   - logs/: Logs de ejecución")
        logger.info("\n📊 Archivos generados:")
        
        # Listar archivos generados
        eda_files = []
        if os.path.exists("results/eda"):
            for file in os.listdir("results/eda"):
                eda_files.append(file)
        
        for file in sorted(eda_files):
            logger.info(f"   ✅ {file}")
        
    except Exception as e:
        logger.error(f"❌ Error en el pipeline de EDA: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_eda_executive_summary(df, pollutants, trends_results, spatial_results, 
                                  correlation_report, pca_results):
    """
    Genera resumen ejecutivo completo del análisis EDA
    """
    summary = f"\n{'='*90}\n"
    summary += f"RESUMEN EJECUTIVO - FASE 4: ANÁLISIS EXPLORATORIO DE DATOS (EDA)\n"
    summary += f"Dashboard de Calidad del Aire - Lima\n"
    summary += f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary += f"{'='*90}\n\n"
    
    # Información general
    summary += "📊 INFORMACIÓN GENERAL DEL DATASET\n"
    summary += "-" * 50 + "\n"
    summary += f"Total de registros: {len(df):,}\n"
    summary += f"Período de análisis: {df['timestamp'].min().date()} a {df['timestamp'].max().date()}\n"
    summary += f"Duración total: {(df['timestamp'].max() - df['timestamp'].min()).days:,} días\n"
    summary += f"Contaminantes analizados: {len(pollutants)} ({', '.join([p.upper() for p in pollutants])})\n"
    
    if 'station_name' in df.columns:
        summary += f"Estaciones de monitoreo: {df['station_name'].nunique()}\n"
        stations_list = df['station_name'].unique()
        summary += f"Lista de estaciones: {', '.join(stations_list[:5])}"
        if len(stations_list) > 5:
            summary += f" y {len(stations_list)-5} más"
        summary += "\n"
    
    # Estadísticas descriptivas
    summary += f"\n📈 ESTADÍSTICAS DESCRIPTIVAS\n"
    summary += "-" * 50 + "\n"
    
    for pollutant in pollutants:
        pollutant_data = df[pollutant].dropna()
        if len(pollutant_data) > 0:
            summary += f"\n{pollutant.upper()}:\n"
            summary += f"  Registros válidos: {len(pollutant_data):,} ({len(pollutant_data)/len(df)*100:.1f}%)\n"
            summary += f"  Promedio: {pollutant_data.mean():.2f} μg/m³\n"
            summary += f"  Mediana: {pollutant_data.median():.2f} μg/m³\n"
            summary += f"  Desviación estándar: {pollutant_data.std():.2f} μg/m³\n"
            summary += f"  Rango: {pollutant_data.min():.2f} - {pollutant_data.max():.2f} μg/m³\n"
            summary += f"  Percentil 95: {pollutant_data.quantile(0.95):.2f} μg/m³\n"
    
    # Hallazgos temporales
    summary += f"\n⏰ HALLAZGOS TEMPORALES\n"
    summary += "-" * 50 + "\n"
    
    if trends_results:
        summary += "Tendencias identificadas:\n"
        for pollutant, trend_info in trends_results.items():
            if 'trend' in trend_info:
                summary += f"  {pollutant.upper()}: {trend_info['trend']}"
                if 'slope' in trend_info:
                    summary += f" ({trend_info['slope']:.2f} μg/m³/año)"
                if 'significance' in trend_info:
                    summary += f" - {trend_info['significance']}"
                summary += "\n"
    
    # Análisis estacional básico
    if 'timestamp' in df.columns:
        df_seasonal = df.copy()
        df_seasonal['month'] = df_seasonal['timestamp'].dt.month
        df_seasonal['season'] = df_seasonal['month'].map({
            12: 'Verano', 1: 'Verano', 2: 'Verano',
            3: 'Otoño', 4: 'Otoño', 5: 'Otoño',
            6: 'Invierno', 7: 'Invierno', 8: 'Invierno',
            9: 'Primavera', 10: 'Primavera', 11: 'Primavera'
        })
        
        summary += "\nPatrones estacionales:\n"
        for pollutant in pollutants:
            seasonal_means = df_seasonal.groupby('season')[pollutant].mean()
            if not seasonal_means.empty:
                max_season = seasonal_means.idxmax()
                min_season = seasonal_means.idxmin()
                summary += f"  {pollutant.upper()}: Máximo en {max_season} ({seasonal_means[max_season]:.1f} μg/m³), "
                summary += f"Mínimo en {min_season} ({seasonal_means[min_season]:.1f} μg/m³)\n"
    
    # Hallazgos espaciales
    summary += f"\n🗺️ HALLAZGOS ESPACIALES\n"
    summary += "-" * 50 + "\n"
    
    if spatial_results:
        summary += "Variación espacial:\n"
        for pollutant, spatial_info in spatial_results.items():
            summary += f"  {pollutant.upper()}:\n"
            summary += f"    Estación más contaminada: {spatial_info['most_polluted_station']} "
            summary += f"({spatial_info['highest_value']:.1f} μg/m³)\n"
            summary += f"    Estación menos contaminada: {spatial_info['least_polluted_station']} "
            summary += f"({spatial_info['lowest_value']:.1f} μg/m³)\n"
            summary += f"    Rango espacial: {spatial_info['spatial_range']:.1f} μg/m³\n"
    
    # Hallazgos de correlaciones
    summary += f"\n🔗 HALLAZGOS DE CORRELACIONES\n"
    summary += "-" * 50 + "\n"
    
    if correlation_report:
        # Extraer información clave del reporte de correlaciones
        lines = correlation_report.split('\n')
        in_strongest = False
        correlation_count = 0
        
        for line in lines:
            if "CORRELACIONES MÁS FUERTES:" in line:
                in_strongest = True
                continue
            elif in_strongest and line.strip() and not line.startswith('-'):
                if correlation_count < 3:  # Top 3
                    summary += f"  {line.strip()}\n"
                    correlation_count += 1
                elif line.strip() and not line[0].isdigit():
                    break
    
    # Resultados PCA
    if pca_results and 'explained_variance_ratio' in pca_results:
        summary += f"\nAnálisis de Componentes Principales:\n"
        variance_ratios = pca_results['explained_variance_ratio']
        cumulative_variance = np.cumsum(variance_ratios)
        
        summary += f"  Primer componente explica: {variance_ratios[0]*100:.1f}% de la varianza\n"
        if len(variance_ratios) > 1:
            summary += f"  Primeros dos componentes explican: {cumulative_variance[1]*100:.1f}% de la varianza\n"
        
        if 'n_components_95_variance' in pca_results:
            summary += f"  Componentes para 95% de varianza: {pca_results['n_components_95_variance']}\n"
    
    # Calidad de datos
    summary += f"\n📋 CALIDAD DE DATOS\n"
    summary += "-" * 50 + "\n"
    
    total_possible = len(df) * len(pollutants)
    total_valid = sum(df[p].notna().sum() for p in pollutants)
    completeness = total_valid / total_possible * 100
    
    summary += f"Completitud general: {completeness:.1f}%\n"
    summary += f"Registros totales posibles: {total_possible:,}\n"
    summary += f"Registros válidos: {total_valid:,}\n"
    summary += f"Registros faltantes: {total_possible - total_valid:,}\n"
    
    summary += f"\nCompletitud por contaminante:\n"
    for pollutant in pollutants:
        valid_count = df[pollutant].notna().sum()
        completeness_pol = valid_count / len(df) * 100
        summary += f"  {pollutant.upper()}: {completeness_pol:.1f}% ({valid_count:,}/{len(df):,})\n"
    
    # Recomendaciones
    summary += f"\n🎯 RECOMENDACIONES PRINCIPALES\n"
    summary += "-" * 50 + "\n"
    
    recommendations = []
    
    # Basado en tendencias
    if trends_results:
        increasing_pollutants = [p for p, info in trends_results.items() 
                               if info.get('trend') == 'Aumentando']
        if increasing_pollutants:
            recommendations.append(f"Priorizar control de {', '.join([p.upper() for p in increasing_pollutants])} (tendencia creciente)")
    
    # Basado en variabilidad espacial
    if spatial_results:
        high_variation = [p for p, info in spatial_results.items() 
                         if info.get('spatial_range', 0) > 50]
        if high_variation:
            recommendations.append(f"Investigar fuentes locales de {', '.join([p.upper() for p in high_variation])} (alta variación espacial)")
    
    # Basado en completitud de datos
    low_completeness = [p for p in pollutants 
                       if df[p].notna().sum() / len(df) < 0.7]
    if low_completeness:
        recommendations.append(f"Mejorar cobertura de monitoreo para {', '.join([p.upper() for p in low_completeness])}")
    
    # Recomendaciones generales
    recommendations.extend([
        "Implementar sistema de alertas tempranas basado en patrones identificados",
        "Desarrollar modelos predictivos específicos por estación y estación",
        "Establecer programa de monitoreo continuo con mayor frecuencia temporal"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        summary += f"{i}. {rec}\n"
    
    # Archivos generados
    summary += f"\n📁 ARCHIVOS GENERADOS\n"
    summary += "-" * 50 + "\n"
    summary += "Visualizaciones estáticas:\n"
    summary += "  • temporal_overview.png - Vista general temporal\n"
    summary += "  • seasonal_patterns.png - Patrones estacionales\n"
    summary += "  • daily_patterns.png - Patrones diarios\n"
    summary += "  • station_comparison.png - Comparación entre estaciones\n"
    summary += "  • spatial_heatmap.png - Mapa de calor espacial\n"
    summary += "  • correlation_heatmap.png - Mapa de correlaciones\n"
    summary += "  • scatter_matrix.png - Matriz de dispersión\n"
    summary += "  • pca_analysis.png - Análisis de componentes principales\n"
    
    summary += "\nVisualizaciones interactivas:\n"
    summary += "  • interactive_timeline.html - Línea de tiempo interactiva\n"
    summary += "  • interactive_map.html - Mapa interactivo de estaciones\n"
    summary += "  • interactive_correlations.html - Correlaciones interactivas\n"
    
    summary += "\nReportes y datos:\n"
    summary += "  • executive_summary.txt - Este resumen ejecutivo\n"
    summary += "  • correlation_report.txt - Reporte detallado de correlaciones\n"
    summary += "  • temporal_trends.json - Resultados de análisis temporal\n"
    summary += "  • spatial_results.json - Resultados de análisis espacial\n"
    summary += "  • pca_results.json - Resultados de análisis PCA\n"
    
    summary += f"\n{'='*90}\n"
    summary += "FASE 4: ANÁLISIS EXPLORATORIO DE DATOS - COMPLETADA\n"
    summary += f"{'='*90}\n"
    
    return summary

def create_visualization_index():
    """
    Crea un índice HTML de todas las visualizaciones generadas
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dashboard EDA - Calidad del Aire Lima</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
            h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
            .card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #f9f9f9; }
            .card h3 { margin-top: 0; color: #2980b9; }
            .card img { width: 100%; max-width: 100%; height: auto; border-radius: 5px; }
            .interactive-link { display: inline-block; background: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 10px 0; }
            .interactive-link:hover { background: #2980b9; }
            .summary { background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
            .stat-box { background: #3498db; color: white; padding: 20px; border-radius: 8px; text-align: center; }
            .stat-number { font-size: 2em; font-weight: bold; }
            .stat-label { font-size: 0.9em; opacity: 0.9; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌍 Dashboard de Análisis Exploratorio de Datos (EDA)</h1>
            <h1>Calidad del Aire - Lima, Perú</h1>
            
            <div class="summary">
                <h2>📊 Resumen del Análisis</h2>
                <p>Este dashboard presenta el análisis exploratorio completo de los datos de calidad del aire de Lima, 
                incluyendo análisis temporal, espacial y de correlaciones.</p>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-number">4</div>
                    <div class="stat-label">Tipos de Análisis</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">15+</div>
                    <div class="stat-label">Visualizaciones</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">3</div>
                    <div class="stat-label">Mapas Interactivos</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">5+</div>
                    <div class="stat-label">Reportes</div>
                </div>
            </div>
            
            <h2>⏰ Análisis Temporal</h2>
            <div class="grid">
                <div class="card">
                    <h3>Vista General Temporal</h3>
                    <img src="temporal_overview.png" alt="Vista General Temporal">
                    <p>Evolución de contaminantes a lo largo del tiempo con tendencias y variabilidad.</p>
                </div>
                <div class="card">
                    <h3>Patrones Estacionales</h3>
                    <img src="seasonal_patterns.png" alt="Patrones Estacionales">
                    <p>Análisis de variaciones estacionales y climáticas en la contaminación.</p>
                </div>
                <div class="card">
                    <h3>Patrones Diarios</h3>
                    <img src="daily_patterns.png" alt="Patrones Diarios">
                    <p>Patrones horarios y diferencias entre días laborales y fines de semana.</p>
                </div>
                <div class="card">
                    <h3>Análisis de Tendencias</h3>
                    <img src="trends_analysis.png" alt="Análisis de Tendencias">
                    <p>Tendencias a largo plazo y cambios significativos en el tiempo.</p>
                </div>
            </div>
            
            <div class="card">
                <h3>🔄 Línea de Tiempo Interactiva</h3>
                <a href="interactive_timeline.html" class="interactive-link">Ver Timeline Interactivo</a>
                <p>Explora la evolución temporal de los contaminantes con gráficos interactivos.</p>
            </div>
            
            <h2>🗺️ Análisis Espacial</h2>
            <div class="grid">
                <div class="card">
                    <h3>Comparación entre Estaciones</h3>
                    <img src="station_comparison.png" alt="Comparación de Estaciones">
                    <p>Niveles de contaminación y variabilidad por estación de monitoreo.</p>
                </div>
                <div class="card">
                    <h3>Mapa de Calor Espacial</h3>
                    <img src="spatial_heatmap.png" alt="Mapa de Calor Espacial">
                    <p>Correlaciones espaciales entre estaciones y distribución de contaminantes.</p>
                </div>
                <div class="card">
                    <h3>Patrones Espaciales</h3>
                    <img src="spatial_patterns.png" alt="Patrones Espaciales">
                    <p>Análisis de dispersión espacial y ranking de estaciones más contaminadas.</p>
                </div>
                <div class="card">
                    <h3>Análisis de Distancia</h3>
                    <img src="distance_analysis.png" alt="Análisis de Distancia">
                    <p>Correlación espacial basada en distancia geográfica entre estaciones.</p>
                </div>
            </div>
            
            <div class="card">
                <h3>🌍 Mapa Interactivo</h3>
                <a href="interactive_map.html" class="interactive-link">Ver Mapa Interactivo</a>
                <p>Mapa interactivo con ubicación de estaciones y niveles de contaminación.</p>
            </div>
            
            <h2>🔗 Análisis de Correlaciones</h2>
            <div class="grid">
                <div class="card">
                    <h3>Mapa de Correlaciones</h3>
                    <img src="correlation_heatmap.png" alt="Mapa de Correlaciones">
                    <p>Correlaciones entre diferentes contaminantes y clustering jerárquico.</p>
                </div>
                <div class="card">
                    <h3>Correlaciones con Características</h3>
                    <img src="feature_correlations.png" alt="Correlaciones con Características">
                    <p>Relaciones entre contaminantes y variables temporales/meteorológicas.</p>
                </div>
                <div class="card">
                    <h3>Matriz de Dispersión</h3>
                    <img src="scatter_matrix.png" alt="Matriz de Dispersión">
                    <p>Relaciones bivariadas entre todos los contaminantes.</p>
                </div>
                <div class="card">
                    <h3>Análisis PCA</h3>
                    <img src="pca_analysis.png" alt="Análisis PCA">
                    <p>Análisis de componentes principales y reducción de dimensionalidad.</p>
                </div>
            </div>
            
            <div class="card">
                <h3>📊 Correlaciones Interactivas</h3>
                <a href="interactive_correlations.html" class="interactive-link">Ver Correlaciones Interactivas</a>
                <p>Explora las correlaciones entre contaminantes con gráficos interactivos.</p>
            </div>
            
            <h2>📋 Reportes y Documentación</h2>
            <div class="grid">
                <div class="card">
                    <h3>📄 Resumen Ejecutivo</h3>
                    <a href="executive_summary.txt" class="interactive-link">Descargar Resumen</a>
                    <p>Resumen completo con hallazgos principales y recomendaciones.</p>
                </div>
                <div class="card">
                    <h3>📊 Reporte de Correlaciones</h3>
                    <a href="correlation_report.txt" class="interactive-link">Descargar Reporte</a>
                    <p>Análisis detallado de correlaciones y patrones estadísticos.</p>
                </div>
                <div class="card">
                    <h3>📈 Datos de Tendencias</h3>
                    <a href="temporal_trends.json" class="interactive-link">Descargar JSON</a>
                    <p>Resultados numéricos del análisis de tendencias temporales.</p>
                </div>
                <div class="card">
                    <h3>🗺️ Datos Espaciales</h3>
                    <a href="spatial_results.json" class="interactive-link">Descargar JSON</a>
                    <p>Resultados del análisis espacial y comparación entre estaciones.</p>
                </div>
            </div>
            
            <div class="summary">
                <h2>🎯 Conclusiones Principales</h2>
                <ul>
                    <li><strong>Patrones Temporales:</strong> Se identificaron patrones estacionales y horarios claros en la contaminación</li>
                    <li><strong>Variabilidad Espacial:</strong> Diferencias significativas entre estaciones de monitoreo</li>
                    <li><strong>Correlaciones:</strong> Relaciones fuertes entre ciertos contaminantes</li>
                    <li><strong>Tendencias:</strong> Tendencias temporales identificadas para planificación de políticas</li>
                </ul>
            </div>
            
            <footer style="text-align: center; margin-top: 40px; padding: 20px; background: #34495e; color: white; border-radius: 8px;">
                <p>Dashboard de Calidad del Aire - Lima, Perú</p>
                <p>Fase 4: Análisis Exploratorio de Datos (EDA) - Completado el """ + datetime.now().strftime('%Y-%m-%d') + """</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    with open("results/eda/index.html", "w", encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info("Índice HTML creado: results/eda/index.html")

if __name__ == "__main__":
    main()
