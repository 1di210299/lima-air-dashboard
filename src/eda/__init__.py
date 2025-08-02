#!/usr/bin/env python3
"""
Módulo EDA (Análisis Exploratorio de Datos) para el Dashboard de Calidad del Aire
Contiene todas las clases y funciones para análisis exploratorio completo
"""

# Importar todas las clases principales del módulo EDA
try:
    from .temporal_analysis import TemporalAnalyzer
    from .spatial_analysis import SpatialAnalyzer
    from .correlation_analysis import CorrelationAnalyzer
    from .pollution_patterns import PollutionPatternAnalyzer
    from .comparative_analysis import ComparativeAnalyzer
    from .interactive_dashboard import InteractiveDashboard
    
    __all__ = [
        'TemporalAnalyzer',
        'SpatialAnalyzer',
        'CorrelationAnalyzer',
        'PollutionPatternAnalyzer',
        'ComparativeAnalyzer',
        'InteractiveDashboard'
    ]
    
    import_success = True
    
except ImportError as e:
    # En caso de que falten dependencias
    print(f"Advertencia: Error importando módulos EDA: {e}")
    __all__ = []
    import_success = False

# Información del módulo
__version__ = '1.0.0'
__author__ = 'Lima Air Quality Dashboard Team'
__description__ = 'Módulo completo de Análisis Exploratorio de Datos para calidad del aire'

def get_available_analyzers():
    """
    Retorna una lista de todos los analizadores disponibles en el módulo
    
    Returns:
        list: Lista de nombres de clases analizadoras disponibles
    """
    return __all__

def create_analyzer(analyzer_type, **kwargs):
    """
    Factory function para crear analizadores específicos
    
    Args:
        analyzer_type (str): Tipo de analizador ('temporal', 'spatial', 'correlation', 'pollution', 'comparative', 'dashboard')
        **kwargs: Argumentos adicionales para el analizador
    
    Returns:
        Instancia del analizador solicitado
    
    Raises:
        ValueError: Si el tipo de analizador no es válido
    """
    if not import_success:
        raise ImportError("Los módulos EDA no se cargaron correctamente")
    
    analyzers = {
        'temporal': TemporalAnalyzer,
        'spatial': SpatialAnalyzer,
        'correlation': CorrelationAnalyzer,
        'pollution': PollutionPatternAnalyzer,
        'patterns': PollutionPatternAnalyzer,
        'comparative': ComparativeAnalyzer,
        'comparison': ComparativeAnalyzer,
        'dashboard': InteractiveDashboard,
        'interactive': InteractiveDashboard
    }
    
    if analyzer_type.lower() not in analyzers:
        raise ValueError(f"Tipo de analizador '{analyzer_type}' no válido. "
                        f"Opciones disponibles: {list(analyzers.keys())}")
    
    return analyzers[analyzer_type.lower()](**kwargs)

def get_analyzer_descriptions():
    """
    Retorna descripciones de cada analizador disponible
    
    Returns:
        dict: Diccionario con descripciones de cada analizador
    """
    descriptions = {
        'TemporalAnalyzer': 'Análisis de patrones temporales, tendencias y estacionalidad',
        'SpatialAnalyzer': 'Análisis de distribución espacial y correlaciones geográficas',
        'CorrelationAnalyzer': 'Análisis de correlaciones entre contaminantes y PCA',
        'PollutionPatternAnalyzer': 'Análisis de patrones de contaminación, episodios y clustering',
        'ComparativeAnalyzer': 'Comparaciones estadísticas entre estaciones y períodos',
        'InteractiveDashboard': 'Dashboard interactivo completo con múltiples visualizaciones'
    }
    return descriptions

# Configuración de logging para el módulo
import logging

# Crear logger específico para el módulo EDA
eda_logger = logging.getLogger('eda')
eda_logger.setLevel(logging.INFO)

# Crear handler si no existe
if not eda_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    eda_logger.addHandler(handler)

# Mensaje de bienvenida del módulo
if import_success:
    eda_logger.info(f"Módulo EDA {__version__} cargado exitosamente")
    eda_logger.info(f"Analizadores disponibles: {', '.join(__all__)}")
else:
    eda_logger.warning("Módulo EDA cargado con errores - verifique dependencias")
