#!/usr/bin/env python3
"""
Dashboard Interactivo - Módulo EDA
Dashboard completo con visualizaciones interactivas y análisis integrado
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import warnings
import logging
from datetime import datetime, timedelta
import json

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class InteractiveDashboard:
    """
    Dashboard interactivo completo para análisis de calidad del aire
    """
    
    def __init__(self):
        """
        Inicializar el dashboard interactivo
        """
        self.data = None
        self.pollutants = ['pm10', 'pm25', 'no2', 'so2', 'o3', 'co']
        self.app = None
        
        # Configuración de colores y estilos
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        # Umbrales WHO
        self.who_standards = {
            'pm10': 20, 'pm25': 5, 'no2': 10, 
            'so2': 40, 'o3': 100, 'co': 4000
        }
    
    def load_data(self, df):
        """
        Cargar datos para el dashboard
        
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
        
        logger.info(f"Datos cargados para dashboard: {len(self.data)} registros")
    
    def create_comprehensive_dashboard(self, save_path=None):
        """
        Crear dashboard interactivo completo con múltiples pestañas
        
        Args:
            save_path (str): Ruta para guardar el dashboard HTML
        """
        if self.data is None:
            logger.error("No hay datos cargados")
            return None
        
        logger.info("Creando dashboard interactivo completo...")
        
        try:
            # Crear figura con pestañas múltiples usando subplots
            from plotly.subplots import make_subplots
            
            # 1. Vista general - Métricas principales
            overview_fig = self._create_overview_section()
            
            # 2. Análisis temporal
            temporal_fig = self._create_temporal_section()
            
            # 3. Análisis espacial
            spatial_fig = self._create_spatial_section()
            
            # 4. Análisis de correlaciones
            correlation_fig = self._create_correlation_section()
            
            # 5. Análisis de calidad del aire
            quality_fig = self._create_air_quality_section()
            
            # Crear dashboard HTML completo
            dashboard_html = self._create_full_html_dashboard(
                overview_fig, temporal_fig, spatial_fig, 
                correlation_fig, quality_fig
            )
            
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(dashboard_html)
                logger.info(f"Dashboard guardado: {save_path}")
            
            return {
                'overview': overview_fig,
                'temporal': temporal_fig,
                'spatial': spatial_fig,
                'correlation': correlation_fig,
                'air_quality': quality_fig
            }
            
        except Exception as e:
            logger.error(f"Error creando dashboard: {str(e)}")
            return None
    
    def _create_overview_section(self):
        """
        Crear sección de vista general con métricas principales
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Resumen de Contaminantes',
                    'Tendencias Recientes (30 días)',
                    'Distribución de Calidad del Aire',
                    'Alertas y Excedencias'
                ),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "pie"}, {"type": "indicator"}]]
            )
            
            # 1. Resumen de contaminantes - valores promedio
            pollutant_means = []
            pollutant_names = []
            
            for pollutant in self.pollutants:
                data_pol = self.data[pollutant].dropna()
                if len(data_pol) > 0:
                    pollutant_means.append(data_pol.mean())
                    pollutant_names.append(pollutant.upper())
            
            if pollutant_means:
                colors = ['red' if pollutant.lower() in self.who_standards and 
                         mean > self.who_standards[pollutant.lower()] 
                         else 'blue' for pollutant, mean in zip(pollutant_names, pollutant_means)]
                
                fig.add_trace(
                    go.Bar(
                        x=pollutant_names,
                        y=pollutant_means,
                        name='Promedio',
                        marker_color=colors,
                        text=[f'{val:.1f}' for val in pollutant_means],
                        textposition='auto'
                    ),
                    row=1, col=1
                )
            
            # 2. Tendencias recientes
            if 'timestamp' in self.data.columns:
                # Últimos 30 días
                recent_data = self.data[self.data['timestamp'] >= 
                                      self.data['timestamp'].max() - timedelta(days=30)]
                
                if len(recent_data) > 0:
                    daily_means = recent_data.groupby(recent_data['timestamp'].dt.date)[self.pollutants[0]].mean()
                    
                    fig.add_trace(
                        go.Scatter(
                            x=daily_means.index,
                            y=daily_means.values,
                            mode='lines+markers',
                            name=f'{self.pollutants[0].upper()} - Últimos 30 días',
                            line=dict(color=self.colors['primary'], width=3)
                        ),
                        row=1, col=2
                    )
            
            # 3. Distribución de calidad del aire (usando PM2.5 como ejemplo)
            if 'pm25' in self.pollutants:
                pm25_data = self.data['pm25'].dropna()
                if len(pm25_data) > 0:
                    # Categorizar según WHO
                    good = (pm25_data <= 5).sum()
                    moderate = ((pm25_data > 5) & (pm25_data <= 15)).sum()
                    poor = ((pm25_data > 15) & (pm25_data <= 25)).sum()
                    very_poor = (pm25_data > 25).sum()
                    
                    fig.add_trace(
                        go.Pie(
                            labels=['Buena', 'Moderada', 'Mala', 'Muy Mala'],
                            values=[good, moderate, poor, very_poor],
                            marker_colors=['green', 'yellow', 'orange', 'red'],
                            name="Calidad del Aire"
                        ),
                        row=2, col=1
                    )
            
            # 4. Indicador de alertas
            total_exceedances = 0
            for pollutant in self.pollutants:
                if pollutant in self.who_standards:
                    pol_data = self.data[pollutant].dropna()
                    if len(pol_data) > 0:
                        exceedances = (pol_data > self.who_standards[pollutant]).sum()
                        total_exceedances += exceedances
            
            exceedance_rate = total_exceedances / (len(self.data) * len(self.pollutants)) * 100
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=exceedance_rate,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "% Excedencias WHO"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "gray"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90}
                    }
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Vista General - Dashboard de Calidad del Aire',
                height=600,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error en sección overview: {str(e)}")
            return None
    
    def _create_temporal_section(self):
        """
        Crear sección de análisis temporal
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Serie Temporal Interactiva',
                    'Patrones Horarios',
                    'Comparación Estacional',
                    'Análisis de Tendencias'
                ),
                specs=[[{"secondary_y": True}, {"type": "bar"}],
                       [{"type": "box"}, {"type": "scatter"}]]
            )
            
            # 1. Serie temporal principal
            if 'timestamp' in self.data.columns:
                for i, pollutant in enumerate(self.pollutants[:3]):  # Top 3
                    daily_data = self.data.groupby(self.data['timestamp'].dt.date)[pollutant].mean()
                    
                    fig.add_trace(
                        go.Scatter(
                            x=daily_data.index,
                            y=daily_data.values,
                            mode='lines',
                            name=pollutant.upper(),
                            line=dict(width=2),
                            visible=True if i == 0 else 'legendonly'
                        ),
                        row=1, col=1
                    )
            
            # 2. Patrones horarios
            if 'timestamp' in self.data.columns:
                hourly_pattern = self.data.groupby(self.data['timestamp'].dt.hour)[self.pollutants[0]].mean()
                
                fig.add_trace(
                    go.Bar(
                        x=list(range(24)),
                        y=hourly_pattern.values,
                        name='Patrón Horario',
                        marker_color=self.colors['secondary']
                    ),
                    row=1, col=2
                )
            
            # 3. Comparación estacional
            if 'timestamp' in self.data.columns:
                seasons = ['Verano', 'Otoño', 'Invierno', 'Primavera']
                season_map = {
                    12: 'Verano', 1: 'Verano', 2: 'Verano',
                    3: 'Otoño', 4: 'Otoño', 5: 'Otoño',
                    6: 'Invierno', 7: 'Invierno', 8: 'Invierno',
                    9: 'Primavera', 10: 'Primavera', 11: 'Primavera'
                }
                
                seasonal_data = self.data.copy()
                seasonal_data['season'] = seasonal_data['timestamp'].dt.month.map(season_map)
                
                for season in seasons:
                    season_values = seasonal_data[seasonal_data['season'] == season][self.pollutants[0]].dropna()
                    if len(season_values) > 0:
                        fig.add_trace(
                            go.Box(
                                y=season_values,
                                name=season,
                                boxpoints='outliers'
                            ),
                            row=2, col=1
                        )
            
            # 4. Análisis de tendencias (regresión)
            if 'timestamp' in self.data.columns and len(self.data) > 30:
                monthly_data = self.data.groupby(self.data['timestamp'].dt.to_period('M'))[self.pollutants[0]].mean()
                
                if len(monthly_data) > 3:
                    x_numeric = np.arange(len(monthly_data))
                    y_values = monthly_data.values
                    
                    # Calcular tendencia
                    z = np.polyfit(x_numeric, y_values, 1)
                    p = np.poly1d(z)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[str(period) for period in monthly_data.index],
                            y=y_values,
                            mode='markers',
                            name='Datos Mensuales',
                            marker=dict(size=8, color=self.colors['primary'])
                        ),
                        row=2, col=2
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[str(period) for period in monthly_data.index],
                            y=p(x_numeric),
                            mode='lines',
                            name='Tendencia',
                            line=dict(color='red', width=3, dash='dash')
                        ),
                        row=2, col=2
                    )
            
            fig.update_layout(
                title='Análisis Temporal - Patrones y Tendencias',
                height=600,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error en sección temporal: {str(e)}")
            return None
    
    def _create_spatial_section(self):
        """
        Crear sección de análisis espacial
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Mapa de Estaciones',
                    'Comparación entre Estaciones',
                    'Ranking de Contaminación',
                    'Correlación Espacial'
                ),
                specs=[[{"type": "scattermapbox"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "heatmap"}]]
            )
            
            # 1. Mapa de estaciones (simulado)
            if 'station_name' in self.data.columns:
                stations = self.data['station_name'].unique()[:10]  # Máximo 10 estaciones
                
                # Simular coordenadas para Lima
                np.random.seed(42)
                lats = -12.0464 + np.random.normal(0, 0.1, len(stations))
                lons = -77.0428 + np.random.normal(0, 0.1, len(stations))
                
                # Calcular concentración promedio por estación
                station_means = []
                for station in stations:
                    station_data = self.data[self.data['station_name'] == station][self.pollutants[0]].mean()
                    station_means.append(station_data)
                
                fig.add_trace(
                    go.Scattermapbox(
                        lat=lats,
                        lon=lons,
                        mode='markers',
                        marker=dict(
                            size=[15 + (mean/max(station_means))*20 for mean in station_means],
                            color=station_means,
                            colorscale='Reds',
                            showscale=True,
                            colorbar=dict(title=f"{self.pollutants[0].upper()} (μg/m³)")
                        ),
                        text=[f"{station}<br>{mean:.1f} μg/m³" for station, mean in zip(stations, station_means)],
                        name='Estaciones'
                    ),
                    row=1, col=1
                )
            
            # 2. Comparación entre estaciones
            if 'station_name' in self.data.columns:
                station_comparison = []
                for station in self.data['station_name'].unique()[:8]:
                    station_data = self.data[self.data['station_name'] == station][self.pollutants[0]].mean()
                    station_comparison.append((station[:15], station_data))
                
                station_comparison.sort(key=lambda x: x[1], reverse=True)
                
                fig.add_trace(
                    go.Bar(
                        x=[s[0] for s in station_comparison],
                        y=[s[1] for s in station_comparison],
                        name='Concentración Promedio',
                        marker_color=[self.colors['danger'] if val > self.who_standards.get(self.pollutants[0], float('inf')) 
                                    else self.colors['primary'] for _, val in station_comparison]
                    ),
                    row=1, col=2
                )
            
            # 3. Ranking de contaminación
            if 'station_name' in self.data.columns:
                ranking_data = []
                for station in self.data['station_name'].unique()[:6]:
                    station_scores = []
                    for pollutant in self.pollutants:
                        pol_mean = self.data[self.data['station_name'] == station][pollutant].mean()
                        if pollutant in self.who_standards and not np.isnan(pol_mean):
                            normalized_score = pol_mean / self.who_standards[pollutant]
                            station_scores.append(normalized_score)
                    
                    if station_scores:
                        overall_score = np.mean(station_scores)
                        ranking_data.append((station[:15], overall_score))
                
                ranking_data.sort(key=lambda x: x[1], reverse=True)
                
                colors = ['red' if score > 1 else 'orange' if score > 0.5 else 'green' 
                         for _, score in ranking_data]
                
                fig.add_trace(
                    go.Bar(
                        y=[s[0] for s in ranking_data],
                        x=[s[1] for s in ranking_data],
                        orientation='h',
                        name='Índice de Contaminación',
                        marker_color=colors
                    ),
                    row=2, col=1
                )
            
            # 4. Correlación espacial (heatmap)
            if 'station_name' in self.data.columns:
                stations = self.data['station_name'].unique()[:6]
                correlation_matrix = np.eye(len(stations))
                
                for i, station1 in enumerate(stations):
                    for j, station2 in enumerate(stations):
                        if i != j:
                            data1 = self.data[self.data['station_name'] == station1][self.pollutants[0]].dropna()
                            data2 = self.data[self.data['station_name'] == station2][self.pollutants[0]].dropna()
                            
                            # Encontrar fechas comunes si hay timestamp
                            if 'timestamp' in self.data.columns and len(data1) > 10 and len(data2) > 10:
                                # Simplificado: usar correlación de medias diarias
                                daily1 = self.data[self.data['station_name'] == station1].groupby(
                                    self.data[self.data['station_name'] == station1]['timestamp'].dt.date)[self.pollutants[0]].mean()
                                daily2 = self.data[self.data['station_name'] == station2].groupby(
                                    self.data[self.data['station_name'] == station2]['timestamp'].dt.date)[self.pollutants[0]].mean()
                                
                                common_dates = set(daily1.index).intersection(set(daily2.index))
                                if len(common_dates) > 5:
                                    vals1 = [daily1[date] for date in common_dates]
                                    vals2 = [daily2[date] for date in common_dates]
                                    correlation = np.corrcoef(vals1, vals2)[0, 1]
                                    correlation_matrix[i, j] = correlation
                
                fig.add_trace(
                    go.Heatmap(
                        z=correlation_matrix,
                        x=[s[:10] for s in stations],
                        y=[s[:10] for s in stations],
                        colorscale='RdBu',
                        zmid=0,
                        name='Correlación'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title='Análisis Espacial - Distribución y Correlaciones',
                height=600,
                mapbox=dict(
                    style='open-street-map',
                    center=dict(lat=-12.0464, lon=-77.0428),
                    zoom=10
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error en sección espacial: {str(e)}")
            return None
    
    def _create_correlation_section(self):
        """
        Crear sección de análisis de correlaciones
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Matriz de Correlación',
                    'Correlaciones Más Fuertes',
                    'Análisis de Componentes Principales',
                    'Red de Correlaciones'
                ),
                specs=[[{"type": "heatmap"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # 1. Matriz de correlación
            corr_data = self.data[self.pollutants].corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_data.values,
                    x=[p.upper() for p in corr_data.columns],
                    y=[p.upper() for p in corr_data.index],
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_data.values, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    name='Correlación'
                ),
                row=1, col=1
            )
            
            # 2. Correlaciones más fuertes
            correlations = []
            for i in range(len(self.pollutants)):
                for j in range(i+1, len(self.pollutants)):
                    corr_val = corr_data.iloc[i, j]
                    if not np.isnan(corr_val):
                        correlations.append({
                            'pair': f"{self.pollutants[i].upper()}-{self.pollutants[j].upper()}",
                            'correlation': abs(corr_val),
                            'original_corr': corr_val
                        })
            
            correlations.sort(key=lambda x: x['correlation'], reverse=True)
            top_correlations = correlations[:6]  # Top 6
            
            colors = ['red' if corr['original_corr'] > 0 else 'blue' for corr in top_correlations]
            
            fig.add_trace(
                go.Bar(
                    x=[corr['pair'] for corr in top_correlations],
                    y=[corr['original_corr'] for corr in top_correlations],
                    name='Correlaciones Fuertes',
                    marker_color=colors
                ),
                row=1, col=2
            )
            
            # 3. PCA
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            pca_data = self.data[self.pollutants].dropna()
            if len(pca_data) > 50:
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(pca_data)
                
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(data_scaled)
                
                fig.add_trace(
                    go.Scatter(
                        x=pca_result[:, 0],
                        y=pca_result[:, 1],
                        mode='markers',
                        name='PCA',
                        marker=dict(size=5, opacity=0.6, color=self.colors['primary']),
                        text=[f"PC1: {pc1:.2f}<br>PC2: {pc2:.2f}" for pc1, pc2 in pca_result[:100]]  # Primeros 100
                    ),
                    row=2, col=1
                )
            
            # 4. Red de correlaciones (scatter plot simulado)
            if len(self.pollutants) >= 2:
                pol1, pol2 = self.pollutants[0], self.pollutants[1]
                scatter_data = self.data[[pol1, pol2]].dropna()
                
                if len(scatter_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=scatter_data[pol1],
                            y=scatter_data[pol2],
                            mode='markers',
                            name=f'{pol1.upper()} vs {pol2.upper()}',
                            marker=dict(size=5, opacity=0.6),
                            text=[f"{pol1}: {x:.1f}<br>{pol2}: {y:.1f}" for x, y in 
                                 zip(scatter_data[pol1][:100], scatter_data[pol2][:100])]
                        ),
                        row=2, col=2
                    )
                    
                    # Línea de tendencia
                    z = np.polyfit(scatter_data[pol1], scatter_data[pol2], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(scatter_data[pol1].min(), scatter_data[pol1].max(), 100)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=p(x_range),
                            mode='lines',
                            name='Tendencia',
                            line=dict(color='red', width=2, dash='dash')
                        ),
                        row=2, col=2
                    )
            
            fig.update_layout(
                title='Análisis de Correlaciones - Relaciones entre Contaminantes',
                height=600,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error en sección correlaciones: {str(e)}")
            return None
    
    def _create_air_quality_section(self):
        """
        Crear sección de análisis de calidad del aire
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Índice de Calidad del Aire',
                    'Excedencias de Estándares WHO',
                    'Distribución de Categorías',
                    'Alertas y Recomendaciones'
                ),
                specs=[[{"type": "indicator"}, {"type": "bar"}],
                       [{"type": "pie"}, {"type": "table"}]]
            )
            
            # 1. Índice de calidad del aire general
            aqi_scores = []
            for pollutant in self.pollutants:
                if pollutant in self.who_standards:
                    pol_data = self.data[pollutant].dropna()
                    if len(pol_data) > 0:
                        avg_concentration = pol_data.mean()
                        who_standard = self.who_standards[pollutant]
                        aqi_score = min(500, (avg_concentration / who_standard) * 100)
                        aqi_scores.append(aqi_score)
            
            overall_aqi = np.mean(aqi_scores) if aqi_scores else 0
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=overall_aqi,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Índice de Calidad del Aire"},
                    gauge={
                        'axis': {'range': [None, 300]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "green"},
                            {'range': [50, 100], 'color': "yellow"},
                            {'range': [100, 150], 'color': "orange"},
                            {'range': [150, 300], 'color': "red"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 100}
                    }
                ),
                row=1, col=1
            )
            
            # 2. Excedencias por contaminante
            exceedance_counts = []
            exceedance_names = []
            
            for pollutant in self.pollutants:
                if pollutant in self.who_standards:
                    pol_data = self.data[pollutant].dropna()
                    if len(pol_data) > 0:
                        exceedances = (pol_data > self.who_standards[pollutant]).sum()
                        exceedance_rate = exceedances / len(pol_data) * 100
                        exceedance_counts.append(exceedance_rate)
                        exceedance_names.append(pollutant.upper())
            
            colors = ['red' if rate > 10 else 'orange' if rate > 5 else 'green' 
                     for rate in exceedance_counts]
            
            fig.add_trace(
                go.Bar(
                    x=exceedance_names,
                    y=exceedance_counts,
                    name='% Excedencias',
                    marker_color=colors,
                    text=[f'{rate:.1f}%' for rate in exceedance_counts],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # 3. Distribución de categorías (usando PM2.5)
            if 'pm25' in self.pollutants:
                pm25_data = self.data['pm25'].dropna()
                if len(pm25_data) > 0:
                    good = (pm25_data <= 5).sum()
                    moderate = ((pm25_data > 5) & (pm25_data <= 15)).sum()
                    poor = ((pm25_data > 15) & (pm25_data <= 25)).sum()
                    very_poor = (pm25_data > 25).sum()
                    
                    fig.add_trace(
                        go.Pie(
                            labels=['Buena (≤5)', 'Moderada (5-15)', 'Mala (15-25)', 'Muy Mala (>25)'],
                            values=[good, moderate, poor, very_poor],
                            marker_colors=['green', 'yellow', 'orange', 'red'],
                            name="Calidad PM2.5"
                        ),
                        row=2, col=1
                    )
            
            # 4. Tabla de alertas y recomendaciones
            alerts = []
            recommendations = []
            
            for pollutant in self.pollutants:
                if pollutant in self.who_standards:
                    pol_data = self.data[pollutant].dropna()
                    if len(pol_data) > 0:
                        current_level = pol_data.iloc[-1] if len(pol_data) > 0 else 0
                        who_standard = self.who_standards[pollutant]
                        
                        if current_level > who_standard * 2:
                            alerts.append(f"ALERTA: {pollutant.upper()} muy elevado")
                            recommendations.append("Evitar actividades al aire libre")
                        elif current_level > who_standard:
                            alerts.append(f"Cuidado: {pollutant.upper()} elevado")
                            recommendations.append("Limitar ejercicio intenso al aire libre")
                        else:
                            alerts.append(f"Normal: {pollutant.upper()} dentro de límites")
                            recommendations.append("Condiciones normales")
            
            # Crear tabla
            table_data = []
            for i in range(min(len(alerts), 6)):  # Máximo 6 filas
                table_data.append([alerts[i], recommendations[i]])
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Estado', 'Recomendación'],
                               fill_color='lightblue',
                               align='left'),
                    cells=dict(values=[[row[0] for row in table_data],
                                     [row[1] for row in table_data]],
                              fill_color='lightgray',
                              align='left')
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Análisis de Calidad del Aire - Estados y Alertas',
                height=600,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error en sección calidad del aire: {str(e)}")
            return None
    
    def _create_full_html_dashboard(self, overview_fig, temporal_fig, spatial_fig, 
                                   correlation_fig, quality_fig):
        """
        Crear dashboard HTML completo con navegación por pestañas
        """
        try:
            # Convertir figuras a HTML
            overview_html = overview_fig.to_html(include_plotlyjs=False, div_id="overview-plot")
            temporal_html = temporal_fig.to_html(include_plotlyjs=False, div_id="temporal-plot")
            spatial_html = spatial_fig.to_html(include_plotlyjs=False, div_id="spatial-plot")
            correlation_html = correlation_fig.to_html(include_plotlyjs=False, div_id="correlation-plot")
            quality_html = quality_fig.to_html(include_plotlyjs=False, div_id="quality-plot")
            
            # Generar estadísticas de resumen
            stats_summary = self._generate_summary_stats()
            
            # HTML completo del dashboard
            dashboard_html = f"""
            <!DOCTYPE html>
            <html lang="es">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Dashboard Interactivo - Calidad del Aire Lima</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 0;
                        background-color: #f8f9fa;
                    }}
                    .header {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 20px;
                        text-align: center;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    .container {{
                        max-width: 1400px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    .stats-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 20px;
                        margin-bottom: 30px;
                    }}
                    .stat-card {{
                        background: white;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        text-align: center;
                    }}
                    .stat-number {{
                        font-size: 2.5em;
                        font-weight: bold;
                        margin-bottom: 10px;
                        color: #667eea;
                    }}
                    .stat-label {{
                        color: #666;
                        font-size: 0.9em;
                    }}
                    .tabs {{
                        display: flex;
                        background: white;
                        border-radius: 10px 10px 0 0;
                        overflow: hidden;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        margin-bottom: 0;
                    }}
                    .tab {{
                        flex: 1;
                        padding: 15px 20px;
                        background: #e9ecef;
                        border: none;
                        cursor: pointer;
                        transition: all 0.3s ease;
                        font-size: 16px;
                        font-weight: 500;
                    }}
                    .tab:hover {{
                        background: #dee2e6;
                    }}
                    .tab.active {{
                        background: white;
                        color: #667eea;
                        border-bottom: 3px solid #667eea;
                    }}
                    .tab-content {{
                        background: white;
                        padding: 20px;
                        border-radius: 0 0 10px 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        min-height: 600px;
                    }}
                    .tab-pane {{
                        display: none;
                    }}
                    .tab-pane.active {{
                        display: block;
                    }}
                    .footer {{
                        background: #343a40;
                        color: white;
                        text-align: center;
                        padding: 20px;
                        margin-top: 40px;
                    }}
                    .alert {{
                        padding: 15px;
                        margin-bottom: 20px;
                        border-radius: 5px;
                        background: #d4edda;
                        border: 1px solid #c3e6cb;
                        color: #155724;
                    }}
                    .alert.warning {{
                        background: #fff3cd;
                        border: 1px solid #ffeaa7;
                        color: #856404;
                    }}
                    .alert.danger {{
                        background: #f8d7da;
                        border: 1px solid #f5c6cb;
                        color: #721c24;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🌍 Dashboard Interactivo de Calidad del Aire</h1>
                    <h2>Lima, Perú - Análisis Completo EDA</h2>
                    <p>Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="container">
                    <!-- Estadísticas Resumen -->
                    <div class="stats-grid">
                        {stats_summary}
                    </div>
                    
                    <!-- Alertas -->
                    <div class="alert">
                        <strong>ℹ️ Información:</strong> Este dashboard presenta un análisis exploratorio completo 
                        de los datos de calidad del aire de Lima. Navega por las pestañas para explorar diferentes aspectos del análisis.
                    </div>
                    
                    <!-- Navegación por pestañas -->
                    <div class="tabs">
                        <button class="tab active" onclick="openTab(event, 'overview')">📊 Vista General</button>
                        <button class="tab" onclick="openTab(event, 'temporal')">⏰ Análisis Temporal</button>
                        <button class="tab" onclick="openTab(event, 'spatial')">🗺️ Análisis Espacial</button>
                        <button class="tab" onclick="openTab(event, 'correlation')">🔗 Correlaciones</button>
                        <button class="tab" onclick="openTab(event, 'quality')">🌬️ Calidad del Aire</button>
                    </div>
                    
                    <div class="tab-content">
                        <!-- Vista General -->
                        <div id="overview" class="tab-pane active">
                            <h3>📊 Vista General del Sistema</h3>
                            <p>Métricas principales y estado actual de la calidad del aire en Lima.</p>
                            {overview_html}
                        </div>
                        
                        <!-- Análisis Temporal -->
                        <div id="temporal" class="tab-pane">
                            <h3>⏰ Análisis Temporal</h3>
                            <p>Patrones temporales, tendencias y variaciones estacionales de los contaminantes.</p>
                            {temporal_html}
                        </div>
                        
                        <!-- Análisis Espacial -->
                        <div id="spatial" class="tab-pane">
                            <h3>🗺️ Análisis Espacial</h3>
                            <p>Distribución geográfica y comparación entre estaciones de monitoreo.</p>
                            {spatial_html}
                        </div>
                        
                        <!-- Correlaciones -->
                        <div id="correlation" class="tab-pane">
                            <h3>🔗 Análisis de Correlaciones</h3>
                            <p>Relaciones entre contaminantes y análisis de componentes principales.</p>
                            {correlation_html}
                        </div>
                        
                        <!-- Calidad del Aire -->
                        <div id="quality" class="tab-pane">
                            <h3>🌬️ Calidad del Aire</h3>
                            <p>Índices de calidad, excedencias de estándares WHO y alertas.</p>
                            {quality_html}
                        </div>
                    </div>
                </div>
                
                <div class="footer">
                    <p>Dashboard de Calidad del Aire - Lima, Perú</p>
                    <p>Análisis Exploratorio de Datos (EDA) - Fase 4 Completada</p>
                    <p>Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <script>
                    function openTab(evt, tabName) {{
                        var i, tabcontent, tablinks;
                        
                        // Ocultar todo el contenido de pestañas
                        tabcontent = document.getElementsByClassName("tab-pane");
                        for (i = 0; i < tabcontent.length; i++) {{
                            tabcontent[i].classList.remove("active");
                        }}
                        
                        // Quitar clase active de todos los botones
                        tablinks = document.getElementsByClassName("tab");
                        for (i = 0; i < tablinks.length; i++) {{
                            tablinks[i].classList.remove("active");
                        }}
                        
                        // Mostrar pestaña actual y activar botón
                        document.getElementById(tabName).classList.add("active");
                        evt.currentTarget.classList.add("active");
                    }}
                    
                    // Configuración de Plotly para mejor interactividad
                    window.PlotlyConfig = {{
                        displayModeBar: true,
                        displaylogo: false,
                        modeBarButtonsToRemove: ['pan2d','lasso2d']
                    }};
                    
                    // Ajustar gráficos al cambiar de pestaña
                    document.addEventListener('DOMContentLoaded', function() {{
                        setTimeout(function() {{
                            window.dispatchEvent(new Event('resize'));
                        }}, 100);
                    }});
                </script>
            </body>
            </html>
            """
            
            return dashboard_html
            
        except Exception as e:
            logger.error(f"Error creando HTML dashboard: {str(e)}")
            return "<html><body><h1>Error generando dashboard</h1></body></html>"
    
    def _generate_summary_stats(self):
        """
        Generar estadísticas de resumen para el dashboard
        """
        try:
            stats_html = ""
            
            # Total de registros
            total_records = len(self.data)
            stats_html += f"""
            <div class="stat-card">
                <div class="stat-number">{total_records:,}</div>
                <div class="stat-label">Total de Registros</div>
            </div>
            """
            
            # Número de estaciones
            if 'station_name' in self.data.columns:
                n_stations = self.data['station_name'].nunique()
                stats_html += f"""
                <div class="stat-card">
                    <div class="stat-number">{n_stations}</div>
                    <div class="stat-label">Estaciones de Monitoreo</div>
                </div>
                """
            
            # Período de datos
            if 'timestamp' in self.data.columns:
                date_range = (self.data['timestamp'].max() - self.data['timestamp'].min()).days
                stats_html += f"""
                <div class="stat-card">
                    <div class="stat-number">{date_range}</div>
                    <div class="stat-label">Días de Datos</div>
                </div>
                """
            
            # Contaminantes monitoreados
            stats_html += f"""
            <div class="stat-card">
                <div class="stat-number">{len(self.pollutants)}</div>
                <div class="stat-label">Contaminantes</div>
            </div>
            """
            
            # Calidad de datos
            total_possible = len(self.data) * len(self.pollutants)
            total_valid = sum(self.data[p].notna().sum() for p in self.pollutants)
            completeness = total_valid / total_possible * 100
            
            stats_html += f"""
            <div class="stat-card">
                <div class="stat-number">{completeness:.1f}%</div>
                <div class="stat-label">Completitud de Datos</div>
            </div>
            """
            
            return stats_html
            
        except Exception as e:
            logger.error(f"Error generando estadísticas: {str(e)}")
            return ""

def main():
    """
    Función principal para pruebas del módulo
    """
    try:
        # Crear datos de prueba
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='H')
        n_stations = 6
        stations = [f'Estación_{i+1}' for i in range(n_stations)]
        
        # Simular datos de contaminación
        np.random.seed(42)
        data_list = []
        
        for date in dates[:2000]:  # Usar subset para pruebas
            for station in stations:
                if np.random.random() > 0.1:  # 90% de datos válidos
                    # Agregar variabilidad realista
                    station_factor = 1 + (hash(station) % 10) / 20
                    hour_factor = 1 + 0.3 * np.sin(2 * np.pi * date.hour / 24)
                    seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
                    
                    row = {
                        'timestamp': date,
                        'station_name': station,
                        'pm10': max(0, np.random.lognormal(3, 0.5) * station_factor * hour_factor * seasonal_factor),
                        'pm25': max(0, np.random.lognormal(2.5, 0.6) * station_factor * hour_factor * seasonal_factor),
                        'no2': max(0, np.random.lognormal(2.8, 0.4) * station_factor * hour_factor),
                        'so2': max(0, np.random.lognormal(1.5, 0.8) * station_factor),
                        'o3': max(0, np.random.lognormal(4, 0.3) * station_factor),
                        'co': max(0, np.random.lognormal(7, 0.4) * station_factor * hour_factor)
                    }
                    data_list.append(row)
        
        df = pd.DataFrame(data_list)
        print(f"Datos de prueba creados: {len(df)} registros")
        
        # Probar el dashboard
        dashboard = InteractiveDashboard()
        dashboard.load_data(df)
        
        print("\nCreando dashboard interactivo completo...")
        dashboard_result = dashboard.create_comprehensive_dashboard("test_dashboard.html")
        
        if dashboard_result:
            print("✅ Dashboard creado exitosamente")
            print("📁 Archivo guardado: test_dashboard.html")
            print("🌐 Abrir en navegador para ver el dashboard interactivo")
        else:
            print("❌ Error creando dashboard")
        
        return dashboard_result
        
    except Exception as e:
        print(f"❌ Error en pruebas: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
