#!/usr/bin/env python3
"""
Análisis de Patrones de Contaminación - Módulo EDA
Análisis especializado de patrones específicos de contaminación
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class PollutionPatternAnalyzer:
    """
    Analizador especializado en patrones de contaminación
    """
    
    def __init__(self):
        """
        Inicializar el analizador de patrones de contaminación
        """
        self.data = None
        self.pollutants = ['pm10', 'pm25', 'no2', 'so2', 'o3', 'co']
        self.who_standards = {
            'pm10': 20,    # μg/m³ - WHO 2021 guidelines
            'pm25': 5,     # μg/m³ - WHO 2021 guidelines  
            'no2': 10,     # μg/m³ - WHO 2021 guidelines
            'so2': 40,     # μg/m³ - WHO 2005 guidelines
            'o3': 100,     # μg/m³ - WHO 2005 guidelines (8-hour mean)
            'co': 4000     # μg/m³ - WHO 2000 guidelines (8-hour mean)
        }
        
        # Configurar estilo de visualización
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def load_data(self, df):
        """
        Cargar datos para análisis
        
        Args:
            df (pd.DataFrame): DataFrame con datos de calidad del aire
        """
        self.data = df.copy()
        
        # Verificar contaminantes disponibles
        available_pollutants = [p for p in self.pollutants if p in df.columns]
        if available_pollutants:
            self.pollutants = available_pollutants
            logger.info(f"Contaminantes disponibles: {self.pollutants}")
        else:
            logger.warning("No se encontraron contaminantes en los datos")
        
        # Preparar datos temporales
        if 'timestamp' in df.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data = self.data.sort_values('timestamp')
        
        logger.info(f"Datos cargados: {len(self.data)} registros")
    
    def analyze_exceedance_patterns(self, save_path=None):
        """
        Analizar patrones de excedencia de estándares WHO
        
        Args:
            save_path (str): Ruta para guardar la visualización
        """
        if self.data is None:
            logger.error("No hay datos cargados")
            return None
        
        logger.info("Analizando patrones de excedencia de estándares WHO...")
        
        try:
            # Crear figura con subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Análisis de Excedencia de Estándares WHO', fontsize=14, fontweight='bold')
            
            # 1. Porcentaje de excedencia por contaminante
            exceedance_rates = {}
            for pollutant in self.pollutants:
                if pollutant in self.who_standards:
                    data_pol = self.data[pollutant].dropna()
                    if len(data_pol) > 0:
                        standard = self.who_standards[pollutant]
                        exceeded = (data_pol > standard).sum()
                        rate = exceeded / len(data_pol) * 100
                        exceedance_rates[pollutant.upper()] = rate
            
            if exceedance_rates:
                ax1 = axes[0, 0]
                pollutants_list = list(exceedance_rates.keys())
                rates = list(exceedance_rates.values())
                colors = plt.cm.Reds([0.5 + 0.5 * (r/max(rates)) for r in rates])
                
                bars = ax1.bar(pollutants_list, rates, color=colors)
                ax1.set_title('Porcentaje de Excedencia de Estándares WHO')
                ax1.set_ylabel('% de Registros que Exceden')
                ax1.set_ylim(0, max(rates) * 1.1 if rates else 100)
                
                # Agregar valores en las barras
                for bar, rate in zip(bars, rates):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                plt.setp(ax1.get_xticklabels(), rotation=45)
            
            # 2. Excedencia temporal por estación
            if 'timestamp' in self.data.columns and 'station_name' in self.data.columns:
                ax2 = axes[0, 1]
                
                # Seleccionar contaminante principal para análisis temporal
                main_pollutant = 'pm25' if 'pm25' in self.pollutants else self.pollutants[0]
                
                if main_pollutant in self.who_standards:
                    # Crear datos mensuales por estación
                    monthly_data = self.data.copy()
                    monthly_data['year_month'] = monthly_data['timestamp'].dt.to_period('M')
                    
                    station_exceedance = []
                    months = []
                    
                    for station in self.data['station_name'].unique()[:5]:  # Top 5 estaciones
                        station_data = monthly_data[monthly_data['station_name'] == station]
                        for period in station_data['year_month'].unique():
                            period_data = station_data[station_data['year_month'] == period][main_pollutant].dropna()
                            if len(period_data) > 0:
                                standard = self.who_standards[main_pollutant]
                                exceedance_rate = (period_data > standard).mean() * 100
                                station_exceedance.append(exceedance_rate)
                                months.append(str(period))
                    
                    if station_exceedance:
                        # Crear heatmap simplificado
                        pivot_data = monthly_data.groupby(['station_name', 'year_month'])[main_pollutant].apply(
                            lambda x: (x > self.who_standards[main_pollutant]).mean() * 100
                        ).reset_index()
                        
                        if len(pivot_data) > 0:
                            # Tomar muestra representativa
                            stations_sample = pivot_data['station_name'].unique()[:5]
                            periods_sample = sorted(pivot_data['year_month'].unique())[-12:]  # Últimos 12 meses
                            
                            heatmap_data = []
                            for station in stations_sample:
                                row = []
                                for period in periods_sample:
                                    value = pivot_data[
                                        (pivot_data['station_name'] == station) & 
                                        (pivot_data['year_month'] == period)
                                    ][main_pollutant]
                                    row.append(value.iloc[0] if len(value) > 0 else 0)
                                heatmap_data.append(row)
                            
                            im = ax2.imshow(heatmap_data, cmap='Reds', aspect='auto')
                            ax2.set_title(f'Excedencia Temporal - {main_pollutant.upper()}')
                            ax2.set_xlabel('Período')
                            ax2.set_ylabel('Estación')
                            ax2.set_xticks(range(len(periods_sample)))
                            ax2.set_xticklabels([str(p)[-7:] for p in periods_sample], rotation=45)
                            ax2.set_yticks(range(len(stations_sample)))
                            ax2.set_yticklabels([s[:15] for s in stations_sample])
                            
                            # Colorbar
                            plt.colorbar(im, ax=ax2, label='% Excedencia')
            
            # 3. Severidad de excedencia
            ax3 = axes[1, 0]
            severity_data = []
            severity_labels = []
            
            for pollutant in self.pollutants:
                if pollutant in self.who_standards:
                    data_pol = self.data[pollutant].dropna()
                    if len(data_pol) > 0:
                        standard = self.who_standards[pollutant]
                        exceeded_data = data_pol[data_pol > standard]
                        
                        if len(exceeded_data) > 0:
                            # Categorías de severidad
                            moderate = exceeded_data[(exceeded_data <= standard * 2)]
                            high = exceeded_data[(exceeded_data > standard * 2) & (exceeded_data <= standard * 5)]
                            extreme = exceeded_data[exceeded_data > standard * 5]
                            
                            severity_data.append([len(moderate), len(high), len(extreme)])
                            severity_labels.append(pollutant.upper())
            
            if severity_data:
                severity_array = np.array(severity_data)
                x = np.arange(len(severity_labels))
                width = 0.25
                
                ax3.bar(x - width, severity_array[:, 0], width, label='Moderada (1-2x)', color='orange')
                ax3.bar(x, severity_array[:, 1], width, label='Alta (2-5x)', color='red')
                ax3.bar(x + width, severity_array[:, 2], width, label='Extrema (>5x)', color='darkred')
                
                ax3.set_title('Severidad de Excedencias')
                ax3.set_xlabel('Contaminante')
                ax3.set_ylabel('Número de Registros')
                ax3.set_xticks(x)
                ax3.set_xticklabels(severity_labels)
                ax3.legend()
            
            # 4. Distribución de concentraciones vs estándares
            ax4 = axes[1, 1]
            
            for i, pollutant in enumerate(self.pollutants[:4]):  # Máximo 4 contaminantes
                if pollutant in self.who_standards:
                    data_pol = self.data[pollutant].dropna()
                    if len(data_pol) > 0:
                        # Densidad de probabilidad
                        density = stats.gaussian_kde(data_pol)
                        x_range = np.linspace(data_pol.min(), data_pol.quantile(0.95), 100)
                        y_density = density(x_range)
                        
                        ax4.plot(x_range, y_density, label=pollutant.upper(), alpha=0.7)
                        
                        # Línea del estándar WHO
                        standard = self.who_standards[pollutant]
                        if standard <= data_pol.quantile(0.95):
                            ax4.axvline(standard, color=plt.cm.tab10(i), linestyle='--', alpha=0.8)
            
            ax4.set_title('Distribución de Concentraciones vs Estándares WHO')
            ax4.set_xlabel('Concentración (μg/m³)')
            ax4.set_ylabel('Densidad de Probabilidad')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Gráfico guardado: {save_path}")
            
            plt.show()
            
            # Retornar resultados numéricos
            return {
                'exceedance_rates': exceedance_rates,
                'who_standards': self.who_standards
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de excedencia: {str(e)}")
            return None
    
    def analyze_pollution_episodes(self, save_path=None):
        """
        Analizar episodios de contaminación (picos y eventos extremos)
        
        Args:
            save_path (str): Ruta para guardar la visualización
        """
        if self.data is None:
            logger.error("No hay datos cargados")
            return None
        
        logger.info("Analizando episodios de contaminación...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Análisis de Episodios de Contaminación', fontsize=14, fontweight='bold')
            
            # 1. Detección de episodios extremos (percentil 95)
            ax1 = axes[0, 0]
            
            episodes = {}
            for pollutant in self.pollutants:
                data_pol = self.data[pollutant].dropna()
                if len(data_pol) > 0:
                    threshold = data_pol.quantile(0.95)
                    extreme_episodes = len(data_pol[data_pol > threshold])
                    episodes[pollutant.upper()] = {
                        'count': extreme_episodes,
                        'threshold': threshold,
                        'percentage': extreme_episodes / len(data_pol) * 100
                    }
            
            if episodes:
                pollutants_list = list(episodes.keys())
                counts = [episodes[p]['count'] for p in pollutants_list]
                
                bars = ax1.bar(pollutants_list, counts, color='darkred', alpha=0.7)
                ax1.set_title('Episodios Extremos por Contaminante\n(Percentil 95)')
                ax1.set_ylabel('Número de Episodios')
                
                # Agregar porcentajes
                for bar, pollutant in zip(bars, pollutants_list):
                    height = bar.get_height()
                    percentage = episodes[pollutant]['percentage']
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                plt.setp(ax1.get_xticklabels(), rotation=45)
            
            # 2. Duración y frecuencia de episodios
            if 'timestamp' in self.data.columns:
                ax2 = axes[0, 1]
                
                # Seleccionar contaminante principal
                main_pollutant = 'pm25' if 'pm25' in self.pollutants else self.pollutants[0]
                
                if main_pollutant in self.data.columns:
                    pol_data = self.data[['timestamp', main_pollutant]].dropna()
                    pol_data = pol_data.sort_values('timestamp')
                    
                    # Definir umbral de episodio
                    threshold = pol_data[main_pollutant].quantile(0.90)
                    pol_data['is_episode'] = pol_data[main_pollutant] > threshold
                    
                    # Encontrar episodios consecutivos
                    pol_data['episode_group'] = (pol_data['is_episode'] != pol_data['is_episode'].shift()).cumsum()
                    
                    episode_durations = []
                    episode_intensities = []
                    
                    for group in pol_data['episode_group'].unique():
                        group_data = pol_data[pol_data['episode_group'] == group]
                        if group_data['is_episode'].iloc[0]:  # Es un episodio
                            duration = len(group_data)
                            intensity = group_data[main_pollutant].mean()
                            episode_durations.append(duration)
                            episode_intensities.append(intensity)
                    
                    if episode_durations:
                        ax2.scatter(episode_durations, episode_intensities, alpha=0.6, s=50)
                        ax2.set_title(f'Duración vs Intensidad de Episodios\n({main_pollutant.upper()})')
                        ax2.set_xlabel('Duración (registros)')
                        ax2.set_ylabel('Intensidad Promedio (μg/m³)')
                        ax2.grid(True, alpha=0.3)
                        
                        # Agregar línea de tendencia
                        if len(episode_durations) > 1:
                            z = np.polyfit(episode_durations, episode_intensities, 1)
                            p = np.poly1d(z)
                            ax2.plot(sorted(episode_durations), p(sorted(episode_durations)), 
                                    "r--", alpha=0.8, linewidth=2)
            
            # 3. Distribución temporal de episodios
            ax3 = axes[1, 0]
            
            if 'timestamp' in self.data.columns:
                # Análisis por hora del día
                hourly_episodes = {}
                
                for pollutant in self.pollutants[:3]:  # Top 3 contaminantes
                    pol_data = self.data[['timestamp', pollutant]].dropna()
                    if len(pol_data) > 0:
                        pol_data['hour'] = pol_data['timestamp'].dt.hour
                        threshold = pol_data[pollutant].quantile(0.90)
                        
                        hourly_count = []
                        for hour in range(24):
                            hour_data = pol_data[pol_data['hour'] == hour]
                            if len(hour_data) > 0:
                                episodes_count = (hour_data[pollutant] > threshold).sum()
                                hourly_count.append(episodes_count)
                            else:
                                hourly_count.append(0)
                        
                        hourly_episodes[pollutant.upper()] = hourly_count
                
                if hourly_episodes:
                    hours = list(range(24))
                    bottom = np.zeros(24)
                    
                    colors = plt.cm.Set3(np.linspace(0, 1, len(hourly_episodes)))
                    
                    for i, (pollutant, counts) in enumerate(hourly_episodes.items()):
                        ax3.bar(hours, counts, bottom=bottom, label=pollutant, 
                               alpha=0.8, color=colors[i])
                        bottom += np.array(counts)
                    
                    ax3.set_title('Distribución Horaria de Episodios')
                    ax3.set_xlabel('Hora del Día')
                    ax3.set_ylabel('Número de Episodios')
                    ax3.set_xticks(range(0, 24, 3))
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
            
            # 4. Correlación entre episodios de diferentes contaminantes
            ax4 = axes[1, 1]
            
            if len(self.pollutants) >= 2:
                # Matriz de co-ocurrencia de episodios
                co_occurrence = np.zeros((len(self.pollutants), len(self.pollutants)))
                
                for i, pol1 in enumerate(self.pollutants):
                    for j, pol2 in enumerate(self.pollutants):
                        if pol1 in self.data.columns and pol2 in self.data.columns:
                            data1 = self.data[pol1].dropna()
                            data2 = self.data[pol2].dropna()
                            
                            if len(data1) > 0 and len(data2) > 0:
                                # Encontrar índices comunes
                                common_idx = self.data[pol1].notna() & self.data[pol2].notna()
                                common_data = self.data[common_idx]
                                
                                if len(common_data) > 0:
                                    threshold1 = data1.quantile(0.90)
                                    threshold2 = data2.quantile(0.90)
                                    
                                    episodes1 = common_data[pol1] > threshold1
                                    episodes2 = common_data[pol2] > threshold2
                                    
                                    co_occurrence[i, j] = (episodes1 & episodes2).sum()
                
                # Crear heatmap
                if co_occurrence.max() > 0:
                    im = ax4.imshow(co_occurrence, cmap='Reds', aspect='auto')
                    ax4.set_title('Co-ocurrencia de Episodios Extremos')
                    ax4.set_xticks(range(len(self.pollutants)))
                    ax4.set_yticks(range(len(self.pollutants)))
                    ax4.set_xticklabels([p.upper() for p in self.pollutants], rotation=45)
                    ax4.set_yticklabels([p.upper() for p in self.pollutants])
                    
                    # Agregar valores en las celdas
                    for i in range(len(self.pollutants)):
                        for j in range(len(self.pollutants)):
                            text = ax4.text(j, i, int(co_occurrence[i, j]),
                                          ha="center", va="center", color="white" if co_occurrence[i, j] > co_occurrence.max()/2 else "black")
                    
                    plt.colorbar(im, ax=ax4, label='Episodios Simultáneos')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Gráfico guardado: {save_path}")
            
            plt.show()
            
            return {
                'episodes': episodes,
                'co_occurrence': co_occurrence.tolist() if 'co_occurrence' in locals() else None
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de episodios: {str(e)}")
            return None
    
    def create_pollution_clustering(self, save_path=None):
        """
        Crear análisis de clustering de patrones de contaminación
        
        Args:
            save_path (str): Ruta para guardar la visualización
        """
        if self.data is None:
            logger.error("No hay datos cargados")
            return None
        
        logger.info("Creando análisis de clustering de contaminación...")
        
        try:
            # Preparar datos para clustering
            cluster_data = self.data[self.pollutants].dropna()
            
            if len(cluster_data) < 50:
                logger.warning("Datos insuficientes para clustering")
                return None
            
            # Normalizar datos
            scaler = StandardScaler()
            data_normalized = scaler.fit_transform(cluster_data)
            
            # Determinar número óptimo de clusters
            inertias = []
            k_range = range(2, min(11, len(cluster_data)//10))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data_normalized)
                inertias.append(kmeans.inertia_)
            
            # Seleccionar número de clusters (método del codo)
            optimal_k = 3  # Default
            if len(inertias) > 2:
                # Buscar el "codo" en la curva
                differences = np.diff(inertias)
                second_differences = np.diff(differences)
                if len(second_differences) > 0:
                    optimal_k = np.argmax(second_differences) + 3  # +3 porque empezamos en k=2
                    optimal_k = min(optimal_k, 6)  # Máximo 6 clusters
            
            # Aplicar clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(data_normalized)
            
            # Crear visualización
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Análisis de Clustering de Patrones de Contaminación', fontsize=14, fontweight='bold')
            
            # 1. Método del codo
            ax1 = axes[0, 0]
            ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
            ax1.axvline(optimal_k, color='red', linestyle='--', label=f'K óptimo = {optimal_k}')
            ax1.set_title('Método del Codo para Determinar K')
            ax1.set_xlabel('Número de Clusters (K)')
            ax1.set_ylabel('Inercia')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Visualización de clusters (PCA 2D)
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=2)
            data_pca = pca.fit_transform(data_normalized)
            
            ax2 = axes[0, 1]
            scatter = ax2.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, 
                                 cmap='viridis', alpha=0.6, s=50)
            ax2.set_title(f'Clusters de Contaminación (K={optimal_k})')
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)')
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)')
            plt.colorbar(scatter, ax=ax2, label='Cluster')
            
            # Agregar centroides
            centroids_pca = pca.transform(kmeans.cluster_centers_)
            ax2.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                       c='red', marker='x', s=200, linewidths=3, label='Centroides')
            ax2.legend()
            
            # 3. Perfil de clusters
            ax3 = axes[1, 0]
            
            cluster_profiles = []
            for i in range(optimal_k):
                cluster_mask = clusters == i
                cluster_profile = cluster_data[cluster_mask].mean()
                cluster_profiles.append(cluster_profile.values)
            
            cluster_profiles = np.array(cluster_profiles)
            
            # Crear heatmap de perfiles
            im = ax3.imshow(cluster_profiles, cmap='RdYlBu_r', aspect='auto')
            ax3.set_title('Perfil Promedio de Clusters')
            ax3.set_xlabel('Contaminante')
            ax3.set_ylabel('Cluster')
            ax3.set_xticks(range(len(self.pollutants)))
            ax3.set_xticklabels([p.upper() for p in self.pollutants], rotation=45)
            ax3.set_yticks(range(optimal_k))
            ax3.set_yticklabels([f'Cluster {i}' for i in range(optimal_k)])
            
            # Agregar valores en las celdas
            for i in range(optimal_k):
                for j in range(len(self.pollutants)):
                    value = cluster_profiles[i, j]
                    ax3.text(j, i, f'{value:.1f}', ha="center", va="center",
                            color="white" if value > cluster_profiles.max()/2 else "black")
            
            plt.colorbar(im, ax=ax3, label='Concentración (μg/m³)')
            
            # 4. Distribución temporal de clusters
            ax4 = axes[1, 1]
            
            if 'timestamp' in self.data.columns:
                # Agregar clusters a datos originales
                cluster_data_with_time = cluster_data.copy()
                cluster_data_with_time['cluster'] = clusters
                cluster_data_with_time['timestamp'] = self.data.loc[cluster_data.index, 'timestamp']
                cluster_data_with_time['hour'] = cluster_data_with_time['timestamp'].dt.hour
                
                # Distribución horaria de clusters
                hourly_cluster_dist = cluster_data_with_time.groupby(['hour', 'cluster']).size().unstack(fill_value=0)
                
                # Normalizar por hora para obtener proporciones
                hourly_cluster_prop = hourly_cluster_dist.div(hourly_cluster_dist.sum(axis=1), axis=0)
                
                # Crear gráfico de barras apiladas
                bottom = np.zeros(24)
                colors = plt.cm.viridis(np.linspace(0, 1, optimal_k))
                
                for i in range(optimal_k):
                    if i in hourly_cluster_prop.columns:
                        values = [hourly_cluster_prop.loc[h, i] if h in hourly_cluster_prop.index else 0 for h in range(24)]
                        ax4.bar(range(24), values, bottom=bottom, label=f'Cluster {i}', 
                               color=colors[i], alpha=0.8)
                        bottom += values
                
                ax4.set_title('Distribución Temporal de Clusters')
                ax4.set_xlabel('Hora del Día')
                ax4.set_ylabel('Proporción de Registros')
                ax4.set_xticks(range(0, 24, 3))
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Gráfico guardado: {save_path}")
            
            plt.show()
            
            # Generar interpretación de clusters
            cluster_interpretation = self._interpret_clusters(cluster_profiles, clusters)
            
            return {
                'optimal_k': optimal_k,
                'cluster_profiles': cluster_profiles.tolist(),
                'cluster_labels': clusters.tolist(),
                'interpretation': cluster_interpretation,
                'inertias': inertias
            }
            
        except Exception as e:
            logger.error(f"Error en clustering: {str(e)}")
            return None
    
    def _interpret_clusters(self, cluster_profiles, clusters):
        """
        Interpretar los clusters encontrados
        """
        interpretation = {}
        
        for i, profile in enumerate(cluster_profiles):
            cluster_size = np.sum(clusters == i)
            cluster_percentage = cluster_size / len(clusters) * 100
            
            # Encontrar contaminantes dominantes
            max_pollutant_idx = np.argmax(profile)
            max_pollutant = self.pollutants[max_pollutant_idx]
            max_value = profile[max_pollutant_idx]
            
            # Clasificar nivel de contaminación
            overall_level = np.mean(profile)
            if overall_level > np.mean(cluster_profiles) * 1.5:
                pollution_level = "Alta"
            elif overall_level < np.mean(cluster_profiles) * 0.5:
                pollution_level = "Baja"
            else:
                pollution_level = "Moderada"
            
            interpretation[f"Cluster {i}"] = {
                'size': int(cluster_size),
                'percentage': round(cluster_percentage, 1),
                'dominant_pollutant': max_pollutant.upper(),
                'dominant_value': round(max_value, 2),
                'pollution_level': pollution_level,
                'description': f"Contaminación {pollution_level.lower()} dominada por {max_pollutant.upper()}"
            }
        
        return interpretation
    
    def create_interactive_pollution_dashboard(self, save_path=None):
        """
        Crear dashboard interactivo de patrones de contaminación
        
        Args:
            save_path (str): Ruta para guardar el dashboard HTML
        """
        if self.data is None:
            logger.error("No hay datos cargados")
            return None
        
        logger.info("Creando dashboard interactivo de patrones de contaminación...")
        
        try:
            # Crear subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Excedencia de Estándares WHO',
                    'Evolución Temporal de Contaminantes',
                    'Distribución de Concentraciones',
                    'Comparación entre Estaciones'
                ),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "histogram"}, {"type": "box"}]]
            )
            
            # 1. Excedencia de estándares WHO
            exceedance_rates = {}
            for pollutant in self.pollutants:
                if pollutant in self.who_standards:
                    data_pol = self.data[pollutant].dropna()
                    if len(data_pol) > 0:
                        standard = self.who_standards[pollutant]
                        exceeded = (data_pol > standard).sum()
                        rate = exceeded / len(data_pol) * 100
                        exceedance_rates[pollutant.upper()] = rate
            
            if exceedance_rates:
                fig.add_trace(
                    go.Bar(
                        x=list(exceedance_rates.keys()),
                        y=list(exceedance_rates.values()),
                        name='Excedencia WHO',
                        marker_color='red',
                        text=[f'{rate:.1f}%' for rate in exceedance_rates.values()],
                        textposition='auto'
                    ),
                    row=1, col=1
                )
            
            # 2. Evolución temporal
            if 'timestamp' in self.data.columns:
                # Datos mensuales
                monthly_data = self.data.copy()
                monthly_data['year_month'] = monthly_data['timestamp'].dt.to_period('M')
                monthly_means = monthly_data.groupby('year_month')[self.pollutants].mean()
                
                for pollutant in self.pollutants[:4]:  # Máximo 4 para legibilidad
                    if pollutant in monthly_means.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=[str(period) for period in monthly_means.index],
                                y=monthly_means[pollutant],
                                mode='lines+markers',
                                name=pollutant.upper(),
                                line=dict(width=2)
                            ),
                            row=1, col=2
                        )
            
            # 3. Distribución de concentraciones
            for pollutant in self.pollutants[:3]:  # Máximo 3 para legibilidad
                data_pol = self.data[pollutant].dropna()
                if len(data_pol) > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=data_pol,
                            name=pollutant.upper(),
                            opacity=0.7,
                            nbinsx=30
                        ),
                        row=2, col=1
                    )
            
            # 4. Comparación entre estaciones (boxplot)
            if 'station_name' in self.data.columns:
                main_pollutant = self.pollutants[0]
                stations = self.data['station_name'].unique()[:5]  # Top 5 estaciones
                
                for station in stations:
                    station_data = self.data[self.data['station_name'] == station][main_pollutant].dropna()
                    if len(station_data) > 0:
                        fig.add_trace(
                            go.Box(
                                y=station_data,
                                name=station[:15],  # Truncar nombre
                                boxpoints='outliers'
                            ),
                            row=2, col=2
                        )
            
            # Actualizar layout
            fig.update_layout(
                title={
                    'text': 'Dashboard Interactivo - Patrones de Contaminación',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                height=800,
                showlegend=True,
                template='plotly_white'
            )
            
            # Actualizar ejes
            fig.update_xaxes(title_text="Contaminante", row=1, col=1)
            fig.update_yaxes(title_text="% Excedencia", row=1, col=1)
            
            fig.update_xaxes(title_text="Período", row=1, col=2)
            fig.update_yaxes(title_text="Concentración (μg/m³)", row=1, col=2)
            
            fig.update_xaxes(title_text="Concentración (μg/m³)", row=2, col=1)
            fig.update_yaxes(title_text="Frecuencia", row=2, col=1)
            
            fig.update_xaxes(title_text="Estación", row=2, col=2)
            fig.update_yaxes(title_text="Concentración (μg/m³)", row=2, col=2)
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Dashboard interactivo guardado: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creando dashboard interactivo: {str(e)}")
            return None

def main():
    """
    Función principal para pruebas del módulo
    """
    try:
        # Crear datos de prueba
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='H')
        n_stations = 5
        stations = [f'Estación_{i+1}' for i in range(n_stations)]
        
        # Simular datos de contaminación
        np.random.seed(42)
        data_list = []
        
        for date in dates[:1000]:  # Usar subset para pruebas
            for station in stations:
                if np.random.random() > 0.1:  # 90% de datos válidos
                    row = {
                        'timestamp': date,
                        'station_name': station,
                        'pm10': max(0, np.random.lognormal(3, 0.5)),
                        'pm25': max(0, np.random.lognormal(2.5, 0.6)),
                        'no2': max(0, np.random.lognormal(2.8, 0.4)),
                        'so2': max(0, np.random.lognormal(1.5, 0.8)),
                        'o3': max(0, np.random.lognormal(4, 0.3)),
                        'co': max(0, np.random.lognormal(7, 0.4))
                    }
                    data_list.append(row)
        
        df = pd.DataFrame(data_list)
        print(f"Datos de prueba creados: {len(df)} registros")
        
        # Probar el analizador
        analyzer = PollutionPatternAnalyzer()
        analyzer.load_data(df)
        
        print("\n1. Análisis de excedencia de estándares WHO...")
        exceedance_results = analyzer.analyze_exceedance_patterns()
        
        print("\n2. Análisis de episodios de contaminación...")
        episodes_results = analyzer.analyze_pollution_episodes()
        
        print("\n3. Análisis de clustering...")
        clustering_results = analyzer.create_pollution_clustering()
        
        print("\n4. Dashboard interactivo...")
        dashboard = analyzer.create_interactive_pollution_dashboard()
        
        print("\n✅ Pruebas completadas exitosamente")
        
        return {
            'exceedance': exceedance_results,
            'episodes': episodes_results,
            'clustering': clustering_results
        }
        
    except Exception as e:
        print(f"❌ Error en pruebas: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
