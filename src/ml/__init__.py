"""
Módulo de Machine Learning para el Dashboard de Calidad del Aire de Lima
Fase 3: Modelado y Predicción ML
"""

# Importar todas las clases principales
try:
    from .data_preparation import DataPreparator
    from .pollution_predictor import PollutionPredictor
    from .timeseries_forecaster import TimeSeriesForecaster
    from .air_quality_classifier import AirQualityClassifier
    
    __all__ = [
        'DataPreparator',
        'PollutionPredictor', 
        'TimeSeriesForecaster',
        'AirQualityClassifier'
    ]
except ImportError as e:
    # En caso de que falten dependencias
    print(f"Advertencia: Error importando módulos ML: {e}")
    __all__ = []

__version__ = "1.0.0"

from .data_preparation import DataPreparator
from .models import PollutionPredictor, TimeSeriesForecaster, AirQualityClassifier
from .model_evaluation import ModelEvaluator
from .model_trainer import ModelTrainer

__all__ = [
    'DataPreparator', 
    'PollutionPredictor', 
    'TimeSeriesForecaster', 
    'AirQualityClassifier',
    'ModelEvaluator',
    'ModelTrainer'
]
