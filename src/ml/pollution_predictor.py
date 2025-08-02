#!/usr/bin/env python3
"""
Modelos de predicción de contaminantes
Fase 3: Modelado y Predicción ML
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PollutionPredictor:
    """
    Clase para predecir niveles de contaminantes usando múltiples algoritmos ML
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = {}
        self.metrics = {}
        
    def train_multiple_models(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             feature_names: List[str]) -> Dict[str, Any]:
        """
        Entrena múltiples modelos y selecciona el mejor
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_test, y_test: Datos de prueba
            feature_names: Nombres de las características
            
        Returns:
            Diccionario con resultados de todos los modelos
        """
        logger.info("🚀 Iniciando entrenamiento de múltiples modelos")
        
        # Definir modelos a entrenar
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=1.0, random_state=42),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, 
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }
        
        results = {}
        
        for model_name, model in models_to_train.items():
            try:
                logger.info(f"Entrenando {model_name}...")
                
                # Entrenar modelo
                model.fit(X_train, y_train)
                
                # Realizar predicciones
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calcular métricas
                metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
                
                # Importancia de características (si está disponible)
                feature_imp = self._get_feature_importance(model, feature_names)
                
                # Guardar resultados
                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'feature_importance': feature_imp,
                    'predictions_train': y_pred_train,
                    'predictions_test': y_pred_test
                }
                
                self.models[model_name] = model
                self.metrics[model_name] = metrics
                self.feature_importance[model_name] = feature_imp
                
                logger.info(f"✅ {model_name} - R²: {metrics['test_r2']:.4f}, "
                           f"RMSE: {metrics['test_rmse']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error entrenando {model_name}: {str(e)}")
                continue
        
        # Seleccionar mejor modelo basado en R² de test
        if results:
            best_model_name = max(results.keys(), 
                                key=lambda x: results[x]['metrics']['test_r2'])
            self.best_model = results[best_model_name]['model']
            self.best_model_name = best_model_name
            
            logger.info(f"🏆 Mejor modelo: {best_model_name} "
                       f"(R² = {results[best_model_name]['metrics']['test_r2']:.4f})")
        
        return results
    
    def predict(self, X: np.ndarray, use_best_model: bool = True, 
                model_name: Optional[str] = None) -> np.ndarray:
        """
        Realiza predicciones usando el mejor modelo o uno específico
        
        Args:
            X: Características para predicción
            use_best_model: Si usar el mejor modelo automáticamente
            model_name: Nombre específico del modelo a usar
            
        Returns:
            Predicciones
        """
        if use_best_model and self.best_model is not None:
            return self.best_model.predict(X)
        elif model_name and model_name in self.models:
            return self.models[model_name].predict(X)
        else:
            raise ValueError("No hay modelo disponible para predicción")
    
    def plot_results(self, results: Dict, y_test: np.ndarray, 
                    save_path: Optional[str] = None) -> None:
        """
        Crea visualizaciones de los resultados
        
        Args:
            results: Resultados de todos los modelos
            y_test: Valores reales de test
            save_path: Ruta para guardar las gráficas
        """
        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Resultados de Modelos de Predicción de Contaminantes', 
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
        
        # Añadir valores en las barras
        for bar, score in zip(bars, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. RMSE por modelo
        ax2 = axes[0, 1]
        bars2 = ax2.bar(x_pos, rmse_scores, alpha=0.8, color='orange')
        ax2.set_xlabel('Modelos')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Comparación de RMSE por Modelo')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, score in zip(bars2, rmse_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_scores)*0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Predicciones vs Reales (mejor modelo)
        ax3 = axes[1, 0]
        if self.best_model_name and self.best_model_name in results:
            best_predictions = results[self.best_model_name]['predictions_test']
            ax3.scatter(y_test, best_predictions, alpha=0.6, s=30)
            
            # Línea perfecta
            min_val = min(np.min(y_test), np.min(best_predictions))
            max_val = max(np.max(y_test), np.max(best_predictions))
            ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax3.set_xlabel('Valores Reales')
            ax3.set_ylabel('Predicciones')
            ax3.set_title(f'Predicciones vs Reales - {self.best_model_name}')
            ax3.grid(True, alpha=0.3)
            
            # Añadir R²
            r2 = results[self.best_model_name]['metrics']['test_r2']
            ax3.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax3.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Importancia de características (mejor modelo)
        ax4 = axes[1, 1]
        if (self.best_model_name and self.best_model_name in results and 
            results[self.best_model_name]['feature_importance'] is not None):
            
            feat_imp = results[self.best_model_name]['feature_importance']
            if len(feat_imp) > 0:
                # Tomar las 10 características más importantes
                feat_imp_sorted = dict(sorted(feat_imp.items(), 
                                            key=lambda x: x[1], reverse=True)[:10])
                
                features = list(feat_imp_sorted.keys())
                importances = list(feat_imp_sorted.values())
                
                y_pos = np.arange(len(features))
                ax4.barh(y_pos, importances, alpha=0.8)
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(features)
                ax4.set_xlabel('Importancia')
                ax4.set_title(f'Top 10 Características - {self.best_model_name}')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfica guardada en: {save_path}")
        
        plt.show()
    
    def plot_feature_importance_comparison(self, results: Dict, 
                                          save_path: Optional[str] = None) -> None:
        """
        Compara la importancia de características entre modelos
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Recopilar importancias de todos los modelos
        all_importances = {}
        for model_name, result in results.items():
            if result['feature_importance'] is not None:
                all_importances[model_name] = result['feature_importance']
        
        if not all_importances:
            logger.warning("No hay importancias de características para mostrar")
            return
        
        # Crear DataFrame con importancias
        feat_df = pd.DataFrame(all_importances).fillna(0)
        
        # Seleccionar top 15 características más importantes en promedio
        feat_df['mean_importance'] = feat_df.mean(axis=1)
        top_features = feat_df.nlargest(15, 'mean_importance')
        
        # Crear heatmap
        sns.heatmap(top_features.drop('mean_importance', axis=1).T, 
                   annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
        ax.set_title('Comparación de Importancia de Características entre Modelos')
        ax.set_ylabel('Modelos')
        ax.set_xlabel('Características')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfica de importancia guardada en: {save_path}")
        
        plt.show()
    
    def save_best_model(self, save_path: str) -> None:
        """Guarda el mejor modelo"""
        if self.best_model is not None:
            joblib.dump({
                'model': self.best_model,
                'model_name': self.best_model_name,
                'metrics': self.metrics[self.best_model_name],
                'feature_importance': self.feature_importance[self.best_model_name]
            }, save_path)
            logger.info(f"Mejor modelo guardado en: {save_path}")
        else:
            logger.error("No hay modelo entrenado para guardar")
    
    def load_model(self, load_path: str) -> None:
        """Carga un modelo guardado"""
        model_data = joblib.load(load_path)
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        logger.info(f"Modelo cargado: {self.best_model_name}")
    
    def _calculate_metrics(self, y_train: np.ndarray, y_pred_train: np.ndarray,
                          y_test: np.ndarray, y_pred_test: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de evaluación"""
        return {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_r2': r2_score(y_test, y_pred_test)
        }
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extrae importancia de características del modelo"""
        try:
            if hasattr(model, 'feature_importances_'):
                # Random Forest, XGBoost, LightGBM, etc.
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Modelos lineales
                importances = np.abs(model.coef_)
            else:
                return None
            
            return dict(zip(feature_names, importances))
        except:
            return None

def main():
    """Función principal para pruebas"""
    try:
        # Importar preparación de datos
        from data_preparation import DataPreparator
        
        # Cargar datos
        df = pd.read_csv("data/lima_air_quality_features.csv")
        logger.info(f"Datos cargados: {len(df):,} registros")
        
        # Preparar datos
        preparator = DataPreparator()
        
        if 'pm10' in df.columns:
            logger.info("🔄 Preparando datos para predicción de PM10...")
            data = preparator.prepare_regression_data(df, 'pm10')
            
            # Crear predictor
            predictor = PollutionPredictor()
            
            # Entrenar modelos
            logger.info("🚀 Entrenando modelos...")
            results = predictor.train_multiple_models(
                data['X_train'], data['y_train'],
                data['X_test'], data['y_test'],
                data['feature_names']
            )
            
            # Crear visualizaciones
            logger.info("📊 Creando visualizaciones...")
            predictor.plot_results(results, data['y_test'], 
                                 save_path="models/pollution_prediction_results.png")
            
            predictor.plot_feature_importance_comparison(
                results, save_path="models/feature_importance_comparison.png"
            )
            
            # Guardar mejor modelo
            predictor.save_best_model("models/best_pollution_model.joblib")
            
            logger.info("🎉 Entrenamiento de modelos completado")
        
        else:
            logger.error("Columna PM10 no encontrada en los datos")
    
    except Exception as e:
        logger.error(f"Error en entrenamiento: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
