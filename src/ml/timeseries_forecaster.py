#!/usr/bin/env python3
"""
Modelos de series temporales para predicción de contaminantes
Fase 3: Modelado y Predicción ML
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# TensorFlow/Keras para redes neuronales
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TensorFlow no disponible. Solo modelos estadísticos.")

# Modelos estadísticos
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesForecaster:
    """
    Clase para pronóstico de series temporales de contaminantes
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}
        self.scalers = {}
        
    def train_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        sequence_length: int) -> Dict[str, Any]:
        """
        Entrena modelo LSTM para series temporales
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_test, y_test: Datos de prueba
            sequence_length: Longitud de la secuencia
            
        Returns:
            Diccionario con resultados del modelo
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow no disponible para entrenar LSTM")
            return {}
        
        logger.info("🧠 Entrenando modelo LSTM...")
        
        # Reshape para LSTM (samples, timesteps, features)
        X_train_lstm = X_train.reshape((X_train.shape[0], sequence_length, 1))
        X_test_lstm = X_test.reshape((X_test.shape[0], sequence_length, 1))
        
        # Construir modelo LSTM
        model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1)
        ])
        
        # Compilar modelo
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=0.001
        )
        
        # Entrenar modelo
        history = model.fit(
            X_train_lstm, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test_lstm, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Predicciones
        y_pred_train = model.predict(X_train_lstm, verbose=0).flatten()
        y_pred_test = model.predict(X_test_lstm, verbose=0).flatten()
        
        # Métricas
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
        
        # Guardar resultados
        result = {
            'model': model,
            'history': history.history,
            'metrics': metrics,
            'predictions_train': y_pred_train,
            'predictions_test': y_pred_test,
            'model_type': 'LSTM'
        }
        
        self.models['LSTM'] = model
        self.metrics['LSTM'] = metrics
        
        logger.info(f"✅ LSTM entrenado - R²: {metrics['test_r2']:.4f}, "
                   f"RMSE: {metrics['test_rmse']:.4f}")
        
        return result
    
    def train_gru_model(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       sequence_length: int) -> Dict[str, Any]:
        """
        Entrena modelo GRU para series temporales
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow no disponible para entrenar GRU")
            return {}
        
        logger.info("🧠 Entrenando modelo GRU...")
        
        # Reshape para GRU
        X_train_gru = X_train.reshape((X_train.shape[0], sequence_length, 1))
        X_test_gru = X_test.reshape((X_test.shape[0], sequence_length, 1))
        
        # Construir modelo GRU
        model = keras.Sequential([
            layers.GRU(50, return_sequences=True, input_shape=(sequence_length, 1)),
            layers.Dropout(0.2),
            layers.GRU(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1)
        ])
        
        # Compilar y entrenar
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        history = model.fit(
            X_train_gru, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test_gru, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Predicciones y métricas
        y_pred_train = model.predict(X_train_gru, verbose=0).flatten()
        y_pred_test = model.predict(X_test_gru, verbose=0).flatten()
        
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
        
        result = {
            'model': model,
            'history': history.history,
            'metrics': metrics,
            'predictions_train': y_pred_train,
            'predictions_test': y_pred_test,
            'model_type': 'GRU'
        }
        
        self.models['GRU'] = model
        self.metrics['GRU'] = metrics
        
        logger.info(f"✅ GRU entrenado - R²: {metrics['test_r2']:.4f}, "
                   f"RMSE: {metrics['test_rmse']:.4f}")
        
        return result
    
    def train_conv1d_model(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          sequence_length: int) -> Dict[str, Any]:
        """
        Entrena modelo Conv1D para series temporales
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow no disponible para entrenar Conv1D")
            return {}
        
        logger.info("🧠 Entrenando modelo Conv1D...")
        
        # Reshape para Conv1D
        X_train_conv = X_train.reshape((X_train.shape[0], sequence_length, 1))
        X_test_conv = X_test.reshape((X_test.shape[0], sequence_length, 1))
        
        # Construir modelo Conv1D
        model = keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', 
                         input_shape=(sequence_length, 1)),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])
        
        # Compilar y entrenar
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        history = model.fit(
            X_train_conv, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test_conv, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Predicciones y métricas
        y_pred_train = model.predict(X_train_conv, verbose=0).flatten()
        y_pred_test = model.predict(X_test_conv, verbose=0).flatten()
        
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
        
        result = {
            'model': model,
            'history': history.history,
            'metrics': metrics,
            'predictions_train': y_pred_train,
            'predictions_test': y_pred_test,
            'model_type': 'Conv1D'
        }
        
        self.models['Conv1D'] = model
        self.metrics['Conv1D'] = metrics
        
        logger.info(f"✅ Conv1D entrenado - R²: {metrics['test_r2']:.4f}, "
                   f"RMSE: {metrics['test_rmse']:.4f}")
        
        return result
    
    def train_statistical_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Entrena modelos estadísticos simples como baseline
        """
        logger.info("📊 Entrenando modelos estadísticos...")
        
        results = {}
        
        # Modelo de persistencia (valor anterior)
        y_pred_train_persistence = np.roll(y_train, 1)
        y_pred_test_persistence = np.roll(y_test, 1)
        
        # Excluir primer valor (no tiene valor anterior)
        metrics_persistence = self._calculate_metrics(
            y_train[1:], y_pred_train_persistence[1:],
            y_test[1:], y_pred_test_persistence[1:]
        )
        
        results['Persistence'] = {
            'metrics': metrics_persistence,
            'predictions_train': y_pred_train_persistence,
            'predictions_test': y_pred_test_persistence,
            'model_type': 'Statistical'
        }
        
        # Modelo de media móvil simple
        window = min(7, len(y_train) // 10)  # Ventana de 7 o 10% de los datos
        y_pred_train_ma = pd.Series(y_train).rolling(window=window, min_periods=1).mean().values
        y_pred_test_ma = pd.Series(y_test).rolling(window=window, min_periods=1).mean().values
        
        metrics_ma = self._calculate_metrics(y_train, y_pred_train_ma, y_test, y_pred_test_ma)
        
        results['Moving Average'] = {
            'metrics': metrics_ma,
            'predictions_train': y_pred_train_ma,
            'predictions_test': y_pred_test_ma,
            'model_type': 'Statistical'
        }
        
        # Regresión Ridge con características de lag
        if X_train.shape[1] > 1:  # Si hay más de una característica
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X_train, y_train)
            
            y_pred_train_ridge = ridge_model.predict(X_train)
            y_pred_test_ridge = ridge_model.predict(X_test)
            
            metrics_ridge = self._calculate_metrics(
                y_train, y_pred_train_ridge, y_test, y_pred_test_ridge
            )
            
            results['Ridge Regression'] = {
                'model': ridge_model,
                'metrics': metrics_ridge,
                'predictions_train': y_pred_train_ridge,
                'predictions_test': y_pred_test_ridge,
                'model_type': 'Statistical'
            }
            
            self.models['Ridge Regression'] = ridge_model
            self.metrics['Ridge Regression'] = metrics_ridge
        
        # Guardar métricas
        for name, result in results.items():
            if name not in self.metrics:
                self.metrics[name] = result['metrics']
        
        logger.info("✅ Modelos estadísticos entrenados")
        
        return results
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        sequence_length: int) -> Dict[str, Any]:
        """
        Entrena todos los modelos disponibles
        """
        logger.info("🚀 Entrenando todos los modelos de series temporales...")
        
        all_results = {}
        
        # Modelos estadísticos (siempre disponibles)
        statistical_results = self.train_statistical_models(X_train, y_train, X_test, y_test)
        all_results.update(statistical_results)
        
        # Modelos de deep learning (si TensorFlow está disponible)
        if TENSORFLOW_AVAILABLE:
            lstm_result = self.train_lstm_model(X_train, y_train, X_test, y_test, sequence_length)
            if lstm_result:
                all_results['LSTM'] = lstm_result
            
            gru_result = self.train_gru_model(X_train, y_train, X_test, y_test, sequence_length)
            if gru_result:
                all_results['GRU'] = gru_result
            
            conv1d_result = self.train_conv1d_model(X_train, y_train, X_test, y_test, sequence_length)
            if conv1d_result:
                all_results['Conv1D'] = conv1d_result
        
        # Seleccionar mejor modelo
        if all_results:
            best_model_name = max(all_results.keys(), 
                                key=lambda x: all_results[x]['metrics']['test_r2'])
            self.best_model_name = best_model_name
            
            if best_model_name in self.models:
                self.best_model = self.models[best_model_name]
            
            logger.info(f"🏆 Mejor modelo: {best_model_name} "
                       f"(R² = {all_results[best_model_name]['metrics']['test_r2']:.4f})")
        
        return all_results
    
    def plot_timeseries_results(self, results: Dict, y_test: np.ndarray,
                               y_train: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None) -> None:
        """
        Crea visualizaciones de los resultados de series temporales
        """
        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        num_models = len(results)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Resultados de Modelos de Series Temporales', 
                     fontsize=16, fontweight='bold')
        
        # 1. Comparación de métricas
        model_names = list(results.keys())
        r2_scores = [results[name]['metrics']['test_r2'] for name in model_names]
        rmse_scores = [results[name]['metrics']['test_rmse'] for name in model_names]
        
        ax1 = axes[0, 0]
        x_pos = np.arange(len(model_names))
        bars = ax1.bar(x_pos, r2_scores, alpha=0.8)
        ax1.set_xlabel('Modelos')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Comparación de R² por Modelo')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        for bar, score in zip(bars, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Comparación de RMSE
        ax2 = axes[0, 1]
        bars2 = ax2.bar(x_pos, rmse_scores, alpha=0.8, color='orange')
        ax2.set_xlabel('Modelos')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Comparación de RMSE por Modelo')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        for bar, score in zip(bars2, rmse_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_scores)*0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Serie temporal del mejor modelo
        ax3 = axes[1, 0]
        if self.best_model_name and self.best_model_name in results:
            best_predictions = results[self.best_model_name]['predictions_test']
            
            # Mostrar últimos 200 puntos para claridad
            n_points = min(200, len(y_test))
            time_range = range(len(y_test) - n_points, len(y_test))
            
            ax3.plot(time_range, y_test[-n_points:], label='Real', alpha=0.8, linewidth=2)
            ax3.plot(time_range, best_predictions[-n_points:], 
                    label=f'Predicción ({self.best_model_name})', alpha=0.8, linewidth=2)
            
            ax3.set_xlabel('Tiempo')
            ax3.set_ylabel('Valor del Contaminante')
            ax3.set_title(f'Serie Temporal - {self.best_model_name}')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Añadir R²
            r2 = results[self.best_model_name]['metrics']['test_r2']
            ax3.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax3.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Residuos del mejor modelo
        ax4 = axes[1, 1]
        if self.best_model_name and self.best_model_name in results:
            best_predictions = results[self.best_model_name]['predictions_test']
            residuals = y_test - best_predictions
            
            ax4.scatter(best_predictions, residuals, alpha=0.6, s=30)
            ax4.axhline(y=0, color='r', linestyle='--', alpha=0.8)
            ax4.set_xlabel('Predicciones')
            ax4.set_ylabel('Residuos')
            ax4.set_title(f'Análisis de Residuos - {self.best_model_name}')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfica guardada en: {save_path}")
        
        plt.show()
    
    def plot_training_history(self, results: Dict, save_path: Optional[str] = None) -> None:
        """
        Muestra el historial de entrenamiento de modelos de deep learning
        """
        dl_models = {name: result for name, result in results.items() 
                    if result.get('history') is not None}
        
        if not dl_models:
            logger.info("No hay modelos de deep learning para mostrar historial")
            return
        
        num_models = len(dl_models)
        fig, axes = plt.subplots(1, num_models, figsize=(5*num_models, 4))
        
        if num_models == 1:
            axes = [axes]
        
        for i, (model_name, result) in enumerate(dl_models.items()):
            history = result['history']
            
            axes[i].plot(history['loss'], label='Train Loss', alpha=0.8)
            if 'val_loss' in history:
                axes[i].plot(history['val_loss'], label='Validation Loss', alpha=0.8)
            
            axes[i].set_xlabel('Época')
            axes[i].set_ylabel('Loss')
            axes[i].set_title(f'Historial de Entrenamiento - {model_name}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Historial de entrenamiento guardado en: {save_path}")
        
        plt.show()
    
    def forecast_future(self, last_sequence: np.ndarray, steps: int = 24) -> np.ndarray:
        """
        Realiza pronóstico hacia el futuro
        
        Args:
            last_sequence: Última secuencia de datos
            steps: Número de pasos hacia adelante
            
        Returns:
            Array con predicciones futuras
        """
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado para pronóstico")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            if TENSORFLOW_AVAILABLE and hasattr(self.best_model, 'predict'):
                # Modelo de deep learning
                pred = self.best_model.predict(
                    current_sequence.reshape(1, -1, 1), verbose=0
                )[0, 0]
            else:
                # Modelo estadístico
                pred = self.best_model.predict(current_sequence.reshape(1, -1))[0]
            
            predictions.append(pred)
            
            # Actualizar secuencia (remover primer valor, añadir predicción)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred
        
        return np.array(predictions)
    
    def save_best_model(self, save_path: str) -> None:
        """Guarda el mejor modelo"""
        if self.best_model is not None:
            if TENSORFLOW_AVAILABLE and hasattr(self.best_model, 'save'):
                # Modelo de TensorFlow
                self.best_model.save(save_path)
            else:
                # Modelo de scikit-learn
                joblib.dump({
                    'model': self.best_model,
                    'model_name': self.best_model_name,
                    'metrics': self.metrics.get(self.best_model_name, {})
                }, save_path)
            
            logger.info(f"Mejor modelo guardado en: {save_path}")
        else:
            logger.error("No hay modelo entrenado para guardar")
    
    def _calculate_metrics(self, y_train: np.ndarray, y_pred_train: np.ndarray,
                          y_test: np.ndarray, y_pred_test: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de evaluación"""
        def safe_r2(y_true, y_pred):
            try:
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                return 1 - (ss_res / (ss_tot + 1e-8))
            except:
                return 0.0
        
        return {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_r2': safe_r2(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_r2': safe_r2(y_test, y_pred_test)
        }

def main():
    """Función principal para pruebas"""
    try:
        # Importar preparación de datos
        sys.path.append(os.path.dirname(__file__))
        from data_preparation import DataPreparator
        
        # Cargar datos
        df = pd.read_csv("data/lima_air_quality_features.csv")
        logger.info(f"Datos cargados: {len(df):,} registros")
        
        # Preparar datos de series temporales
        preparator = DataPreparator()
        
        if 'pm10' in df.columns:
            logger.info("🔄 Preparando datos de series temporales para PM10...")
            data = preparator.prepare_timeseries_data(df, 'pm10', sequence_length=24)
            
            # Crear forecaster
            forecaster = TimeSeriesForecaster()
            
            # Entrenar todos los modelos
            logger.info("🚀 Entrenando modelos de series temporales...")
            results = forecaster.train_all_models(
                data['X_train'], data['y_train'],
                data['X_test'], data['y_test'],
                data['sequence_length']
            )
            
            # Crear visualizaciones
            logger.info("📊 Creando visualizaciones...")
            forecaster.plot_timeseries_results(
                results, data['y_test'],
                save_path="models/timeseries_results.png"
            )
            
            if TENSORFLOW_AVAILABLE:
                forecaster.plot_training_history(
                    results, save_path="models/training_history.png"
                )
            
            # Guardar mejor modelo
            forecaster.save_best_model("models/best_timeseries_model")
            
            logger.info("🎉 Entrenamiento de series temporales completado")
        
        else:
            logger.error("Columna PM10 no encontrada en los datos")
    
    except Exception as e:
        logger.error(f"Error en entrenamiento de series temporales: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
