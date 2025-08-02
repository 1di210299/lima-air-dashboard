#!/usr/bin/env python3
"""
Clasificador de calidad del aire
Fase 3: Modelado y Predicción ML
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AirQualityClassifier:
    """
    Clase para clasificar la calidad del aire en categorías
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = {}
        self.metrics = {}
        self.class_names = []
        
    def train_multiple_classifiers(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray, y_test: np.ndarray,
                                  feature_names: List[str], 
                                  class_names: List[str]) -> Dict[str, Any]:
        """
        Entrena múltiples clasificadores y selecciona el mejor
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_test, y_test: Datos de prueba
            feature_names: Nombres de las características
            class_names: Nombres de las clases
            
        Returns:
            Diccionario con resultados de todos los clasificadores
        """
        logger.info("🚀 Iniciando entrenamiento de múltiples clasificadores")
        
        self.class_names = class_names
        
        # Definir clasificadores a entrenar
        classifiers_to_train = {
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            ),
            'Naive Bayes': GaussianNB()
        }
        
        results = {}
        
        for classifier_name, classifier in classifiers_to_train.items():
            try:
                logger.info(f"Entrenando {classifier_name}...")
                
                # Entrenar clasificador
                classifier.fit(X_train, y_train)
                
                # Realizar predicciones
                y_pred_train = classifier.predict(X_train)
                y_pred_test = classifier.predict(X_test)
                
                # Probabilidades (si están disponibles)
                if hasattr(classifier, 'predict_proba'):
                    y_proba_train = classifier.predict_proba(X_train)
                    y_proba_test = classifier.predict_proba(X_test)
                else:
                    y_proba_train = y_proba_test = None
                
                # Calcular métricas
                metrics = self._calculate_classification_metrics(
                    y_train, y_pred_train, y_test, y_pred_test,
                    y_proba_train, y_proba_test
                )
                
                # Validación cruzada
                cv_scores = cross_val_score(classifier, X_train, y_train, 
                                          cv=5, scoring='accuracy', n_jobs=-1)
                metrics['cv_accuracy_mean'] = cv_scores.mean()
                metrics['cv_accuracy_std'] = cv_scores.std()
                
                # Importancia de características (si está disponible)
                feature_imp = self._get_feature_importance(classifier, feature_names)
                
                # Guardar resultados
                results[classifier_name] = {
                    'classifier': classifier,
                    'metrics': metrics,
                    'feature_importance': feature_imp,
                    'predictions_train': y_pred_train,
                    'predictions_test': y_pred_test,
                    'probabilities_train': y_proba_train,
                    'probabilities_test': y_proba_test,
                    'cv_scores': cv_scores
                }
                
                self.models[classifier_name] = classifier
                self.metrics[classifier_name] = metrics
                self.feature_importance[classifier_name] = feature_imp
                
                logger.info(f"✅ {classifier_name} - Accuracy: {metrics['test_accuracy']:.4f}, "
                           f"F1: {metrics['test_f1_weighted']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error entrenando {classifier_name}: {str(e)}")
                continue
        
        # Seleccionar mejor clasificador basado en accuracy de test
        if results:
            best_classifier_name = max(results.keys(), 
                                     key=lambda x: results[x]['metrics']['test_accuracy'])
            self.best_model = results[best_classifier_name]['classifier']
            self.best_model_name = best_classifier_name
            
            logger.info(f"🏆 Mejor clasificador: {best_classifier_name} "
                       f"(Accuracy = {results[best_classifier_name]['metrics']['test_accuracy']:.4f})")
        
        return results
    
    def predict(self, X: np.ndarray, use_best_model: bool = True, 
                model_name: Optional[str] = None) -> np.ndarray:
        """
        Realiza predicciones de clasificación
        
        Args:
            X: Características para predicción
            use_best_model: Si usar el mejor modelo automáticamente
            model_name: Nombre específico del modelo a usar
            
        Returns:
            Predicciones de clase
        """
        if use_best_model and self.best_model is not None:
            return self.best_model.predict(X)
        elif model_name and model_name in self.models:
            return self.models[model_name].predict(X)
        else:
            raise ValueError("No hay clasificador disponible para predicción")
    
    def predict_proba(self, X: np.ndarray, use_best_model: bool = True, 
                     model_name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Realiza predicciones de probabilidad
        """
        model = None
        if use_best_model and self.best_model is not None:
            model = self.best_model
        elif model_name and model_name in self.models:
            model = self.models[model_name]
        
        if model and hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            return None
    
    def plot_classification_results(self, results: Dict, y_test: np.ndarray,
                                   save_path: Optional[str] = None) -> None:
        """
        Crea visualizaciones de los resultados de clasificación
        """
        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Resultados de Clasificación de Calidad del Aire', 
                     fontsize=16, fontweight='bold')
        
        # 1. Comparación de métricas de accuracy
        model_names = list(results.keys())
        accuracy_scores = [results[name]['metrics']['test_accuracy'] for name in model_names]
        f1_scores = [results[name]['metrics']['test_f1_weighted'] for name in model_names]
        
        ax1 = axes[0, 0]
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, accuracy_scores, width, 
                       label='Accuracy', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, f1_scores, width, 
                       label='F1-Score', alpha=0.8)
        
        ax1.set_xlabel('Clasificadores')
        ax1.set_ylabel('Score')
        ax1.set_title('Comparación de Métricas por Clasificador')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, score in zip(bars1, accuracy_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Matriz de confusión del mejor modelo
        ax2 = axes[0, 1]
        if self.best_model_name and self.best_model_name in results:
            best_predictions = results[self.best_model_name]['predictions_test']
            cm = confusion_matrix(y_test, best_predictions)
            
            # Normalizar matriz de confusión
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names,
                       ax=ax2)
            ax2.set_title(f'Matriz de Confusión - {self.best_model_name}')
            ax2.set_xlabel('Predicción')
            ax2.set_ylabel('Real')
        
        # 3. Distribución de clases predichas vs reales
        ax3 = axes[1, 0]
        if self.best_model_name and self.best_model_name in results:
            best_predictions = results[self.best_model_name]['predictions_test']
            
            # Contar distribuciones
            real_counts = np.bincount(y_test)
            pred_counts = np.bincount(best_predictions)
            
            # Asegurar que ambos tengan la misma longitud
            max_len = max(len(real_counts), len(pred_counts))
            real_counts = np.pad(real_counts, (0, max_len - len(real_counts)))
            pred_counts = np.pad(pred_counts, (0, max_len - len(pred_counts)))
            
            x_pos = np.arange(len(self.class_names))
            width = 0.35
            
            ax3.bar(x_pos - width/2, real_counts[:len(self.class_names)], width, 
                   label='Real', alpha=0.8)
            ax3.bar(x_pos + width/2, pred_counts[:len(self.class_names)], width, 
                   label='Predicción', alpha=0.8)
            
            ax3.set_xlabel('Clases de Calidad del Aire')
            ax3.set_ylabel('Cantidad')
            ax3.set_title('Distribución de Clases: Real vs Predicción')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(self.class_names, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
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
    
    def plot_roc_curves(self, results: Dict, y_test: np.ndarray,
                       save_path: Optional[str] = None) -> None:
        """
        Crea curvas ROC para clasificación multiclase
        """
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        from itertools import cycle
        
        # Binarizar las etiquetas para ROC multiclase
        y_test_bin = label_binarize(y_test, classes=range(len(self.class_names)))
        n_classes = y_test_bin.shape[1]
        
        plt.figure(figsize=(12, 8))
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
        
        for model_name, result in results.items():
            if result['probabilities_test'] is not None:
                y_proba = result['probabilities_test']
                
                # Calcular ROC para cada clase
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(n_classes):
                    if i < y_proba.shape[1]:  # Verificar que la clase existe
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Promedio de AUC
                mean_auc = np.mean(list(roc_auc.values()))
                
                # Plotear solo la curva promedio para claridad
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                
                color = next(colors)
                plt.plot([], [], color=color, 
                        label=f'{model_name} (AUC promedio = {mean_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curvas ROC - Clasificación de Calidad del Aire')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Curvas ROC guardadas en: {save_path}")
        
        plt.show()
    
    def generate_classification_report(self, results: Dict, y_test: np.ndarray) -> str:
        """
        Genera reporte detallado de clasificación
        """
        if not self.best_model_name or self.best_model_name not in results:
            return "No hay mejor modelo disponible"
        
        best_result = results[self.best_model_name]
        y_pred = best_result['predictions_test']
        
        report = f"\n{'='*60}\n"
        report += f"REPORTE DE CLASIFICACIÓN DE CALIDAD DEL AIRE\n"
        report += f"{'='*60}\n\n"
        
        report += f"Mejor Modelo: {self.best_model_name}\n"
        report += f"Accuracy: {best_result['metrics']['test_accuracy']:.4f}\n"
        report += f"F1-Score (weighted): {best_result['metrics']['test_f1_weighted']:.4f}\n"
        report += f"Precision (weighted): {best_result['metrics']['test_precision_weighted']:.4f}\n"
        report += f"Recall (weighted): {best_result['metrics']['test_recall_weighted']:.4f}\n\n"
        
        # Reporte detallado por clase
        class_report = classification_report(y_test, y_pred, 
                                           target_names=self.class_names)
        report += "Métricas por Clase:\n"
        report += class_report + "\n"
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        report += "Matriz de Confusión:\n"
        report += str(cm) + "\n\n"
        
        return report
    
    def save_best_model(self, save_path: str) -> None:
        """Guarda el mejor clasificador"""
        if self.best_model is not None:
            joblib.dump({
                'classifier': self.best_model,
                'model_name': self.best_model_name,
                'metrics': self.metrics[self.best_model_name],
                'feature_importance': self.feature_importance[self.best_model_name],
                'class_names': self.class_names
            }, save_path)
            logger.info(f"Mejor clasificador guardado en: {save_path}")
        else:
            logger.error("No hay clasificador entrenado para guardar")
    
    def load_model(self, load_path: str) -> None:
        """Carga un clasificador guardado"""
        model_data = joblib.load(load_path)
        self.best_model = model_data['classifier']
        self.best_model_name = model_data['model_name']
        self.class_names = model_data['class_names']
        logger.info(f"Clasificador cargado: {self.best_model_name}")
    
    def _calculate_classification_metrics(self, y_train: np.ndarray, y_pred_train: np.ndarray,
                                        y_test: np.ndarray, y_pred_test: np.ndarray,
                                        y_proba_train: Optional[np.ndarray] = None,
                                        y_proba_test: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calcula métricas de clasificación"""
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'train_precision_weighted': precision_score(y_train, y_pred_train, average='weighted'),
            'train_recall_weighted': recall_score(y_train, y_pred_train, average='weighted'),
            'train_f1_weighted': f1_score(y_train, y_pred_train, average='weighted'),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision_weighted': precision_score(y_test, y_pred_test, average='weighted'),
            'test_recall_weighted': recall_score(y_test, y_pred_test, average='weighted'),
            'test_f1_weighted': f1_score(y_test, y_pred_test, average='weighted'),
        }
        
        # Métricas macro
        metrics['test_precision_macro'] = precision_score(y_test, y_pred_test, average='macro')
        metrics['test_recall_macro'] = recall_score(y_test, y_pred_test, average='macro')
        metrics['test_f1_macro'] = f1_score(y_test, y_pred_test, average='macro')
        
        # AUC si hay probabilidades y es clasificación binaria o multiclase
        if y_proba_test is not None:
            try:
                if len(np.unique(y_test)) == 2:
                    # Clasificación binaria
                    metrics['test_auc'] = roc_auc_score(y_test, y_proba_test[:, 1])
                else:
                    # Clasificación multiclase
                    metrics['test_auc'] = roc_auc_score(y_test, y_proba_test, 
                                                      multi_class='ovr', average='weighted')
            except:
                pass
        
        return metrics
    
    def _get_feature_importance(self, classifier: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extrae importancia de características del clasificador"""
        try:
            if hasattr(classifier, 'feature_importances_'):
                # Random Forest, XGBoost, LightGBM, etc.
                importances = classifier.feature_importances_
            elif hasattr(classifier, 'coef_'):
                # Modelos lineales (puede ser 2D para multiclase)
                coef = classifier.coef_
                if coef.ndim > 1:
                    # Tomar promedio absoluto para multiclase
                    importances = np.mean(np.abs(coef), axis=0)
                else:
                    importances = np.abs(coef)
            else:
                return None
            
            return dict(zip(feature_names, importances))
        except:
            return None

def main():
    """Función principal para pruebas"""
    try:
        # Importar preparación de datos
        sys.path.append(os.path.dirname(__file__))
        from data_preparation import DataPreparator
        
        # Cargar datos
        df = pd.read_csv("data/lima_air_quality_features.csv")
        logger.info(f"Datos cargados: {len(df):,} registros")
        
        # Preparar datos para clasificación
        preparator = DataPreparator()
        
        if 'pm10' in df.columns:
            logger.info("🔄 Preparando datos para clasificación de calidad del aire...")
            data = preparator.prepare_classification_data(df, 'pm10')
            
            # Crear clasificador
            classifier = AirQualityClassifier()
            
            # Entrenar clasificadores
            logger.info("🚀 Entrenando clasificadores...")
            results = classifier.train_multiple_classifiers(
                data['X_train'], data['y_train'],
                data['X_test'], data['y_test'],
                data['feature_names'], data['class_names']
            )
            
            # Crear visualizaciones
            logger.info("📊 Creando visualizaciones...")
            classifier.plot_classification_results(
                results, data['y_test'],
                save_path="models/classification_results.png"
            )
            
            classifier.plot_roc_curves(
                results, data['y_test'],
                save_path="models/roc_curves.png"
            )
            
            # Generar reporte
            report = classifier.generate_classification_report(results, data['y_test'])
            print(report)
            
            # Guardar reporte
            with open("models/classification_report.txt", "w") as f:
                f.write(report)
            
            # Guardar mejor clasificador
            classifier.save_best_model("models/best_classifier.joblib")
            
            logger.info("🎉 Entrenamiento de clasificadores completado")
        
        else:
            logger.error("Columna PM10 no encontrada en los datos")
    
    except Exception as e:
        logger.error(f"Error en entrenamiento de clasificadores: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
