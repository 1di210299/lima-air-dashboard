# Lima Air Quality Dashboard

## 🎯 Descripción del Proyecto

Dashboard completo para el monitoreo de la calidad del aire en Lima, Perú. Sistema que integra datos históricos del gobierno peruano con información en tiempo real de OpenAQ API para proporcionar análisis predictivo y alertas tempranas.

## 📊 Estado del Proyecto

### ✅ Fase 1: Ingesta y Almacenamiento de Datos (COMPLETADA)
- **Base de datos configurada**: PostgreSQL para producción, SQLite para desarrollo
- **ETL Pipeline funcionando**: Procesamiento automático de grandes volúmenes de datos
- **Sistema de descarga automática**: Integración con OpenAQ API y datos del gobierno
- **Validación de datos**: Detección de outliers y filtrado de datos inválidos
- **Logging completo**: Seguimiento de todos los procesos de ingesta

#### Resultados Obtenidos:
- **Dataset principal**: 927,999 registros consolidados (2010-2024)
- **Última ejecución ETL**: 208,545 registros procesados en 46.39 segundos
- **Tasa de éxito**: 99.5% (207,589 insertados + 956 actualizados)
- **Estaciones activas**: 4 estaciones registradas y funcionando

### ✅ Fase 2: Preprocesamiento y Limpieza (COMPLETADA)
- **Pipeline de limpieza avanzada**: Implementado y funcionando
- **Agregaciones temporales**: Series diarias, semanales y mensuales
- **Features derivadas**: Variables de estacionalidad y tendencias
- **Dataset de entrenamiento**: Listo para modelos ML
- **Normalización para ML**: Datos preparados y escalados

### ✅ Fase 3: Modelado y Predicción ML (COMPLETADA)
- **Modelos de regresión entrenados**: 
  - PM10: Gradient Boosting (R² = 0.6874)
  - NO2: Random Forest (R² = 0.6523)
- **Modelos de clasificación**: Calidad del aire por categorías
- **Evaluación completa**: Métricas MAE/RMSE, gráficos de rendimiento
- **Modelos guardados**: Listos para inferencia en producción
- **Reportes generados**: Análisis completo en `/results/`

### ✅ Fase 5: Dashboard Web (COMPLETADA)
- **Frontend React + TypeScript**: Interfaz moderna y responsiva
- **Componentes implementados**:
  - `AirQualityMap.tsx`: Mapa interactivo con estaciones
  - `TimeSeriesChart.tsx`: Gráficos de series temporales
  - `RunningRiskWidget.tsx`: Widget de riesgo para corredores
- **Integración de mapas**: Leaflet para visualización geográfica
- **Diseño UX/UI**: Interfaz profesional y usable

### 🚧 Fase 4: API Backend (EN DESARROLLO)
- **Estado actual**: Estructura creada, endpoints pendientes
- **Falta implementar**:
  - `GET /current?district=XXX` → medición actual + status
  - `GET /forecast?district=XXX` → pronóstico 48h
  - `GET /risk?district=XXX&age=YY&condition=ZZ` → riesgo personalizado
- **Modelos ML**: Listos para integración en API
- **Documentación**: Swagger/OpenAPI pendiente

### ⏳ Siguientes Fases Pendientes:
6. **Sistema de notificaciones** (Twilio, WhatsApp, alertas)
7. **Despliegue y producción** (Docker, cloud, monitoreo)

## 🏗️ Arquitectura del Sistema

```
lima-air-dashboard/
├── src/
│   ├── database/          # ✅ Modelos y conexiones de BD
│   │   ├── models.py      # ✅ Esquemas SQLAlchemy
│   │   └── connection.py  # ✅ Gestión de conexiones
│   ├── data_ingestion/    # ✅ Pipeline de ingesta
│   │   ├── etl.py         # ✅ Procesamiento ETL
│   │   └── downloader.py  # ✅ Descarga automática
│   ├── preprocessing/     # ✅ Limpieza y transformación
│   │   ├── data_cleaner.py    # ✅ Validación y limpieza
│   │   ├── aggregators.py     # ✅ Series temporales
│   │   └── feature_engineer.py # ✅ Variables derivadas
│   ├── ml/               # ✅ Modelos de Machine Learning
│   │   ├── pollution_predictor.py      # ✅ Regresión
│   │   ├── air_quality_classifier.py  # ✅ Clasificación
│   │   └── timeseries_forecaster.py   # ✅ Pronósticos
│   ├── api/              # 🚧 API REST (En desarrollo)
│   │   └── __init__.py   # ❌ Endpoints pendientes
│   └── notifications/    # ❌ Sistema de alertas (Pendiente)
├── lima-air-dashboard-frontend/  # ✅ Dashboard React
│   ├── src/components/   # ✅ Componentes React
│   │   ├── AirQualityMap.tsx      # ✅ Mapa interactivo
│   │   ├── TimeSeriesChart.tsx    # ✅ Gráficos temporales
│   │   └── RunningRiskWidget.tsx  # ✅ Widget de riesgo
│   └── src/services/     # ✅ Servicios de datos
├── models/               # ✅ Modelos ML entrenados
│   ├── best_pm10_predictor.joblib    # ✅ Predicción PM10
│   ├── best_no2_predictor.joblib     # ✅ Predicción NO2
│   ├── best_pm10_classifier.joblib   # ✅ Clasificación PM10
│   └── best_no2_classifier.joblib    # ✅ Clasificación NO2
├── results/              # ✅ Reportes y métricas ML
│   ├── ml_summary_report.txt         # ✅ Resumen modelos
│   ├── *_results.png                 # ✅ Gráficos rendimiento
│   └── eda/                          # ✅ Análisis exploratorio
├── data/                 # ✅ Datasets procesados
│   ├── lima_air_quality_complete.csv # ✅ Dataset principal (927K registros)
│   ├── lima_air_quality_features.csv # ✅ Features para ML
│   ├── lima_air_quality.db          # ✅ Base de datos SQLite
│   └── aggregated/                   # ✅ Series temporales
├── config/
│   └── settings.py       # ✅ Configuración del sistema
├── logs/                 # ✅ Logs del sistema
└── requirements.txt      # ✅ Dependencias
```

## 🔧 Instalación y Configuración

### Prerrequisitos
```bash
Python 3.11+
PostgreSQL (opcional, usa SQLite por defecto)
```

### Instalación
```bash
# Clonar repositorio (cuando esté en Git)
cd lima-air-dashboard

# Instalar dependencias
pip install -r requirements.txt

# Configurar base de datos
python -c "from src.database.connection import DatabaseManager; DatabaseManager().init_database()"
```

## 🚀 Uso del Sistema

### Ejecución Manual del ETL
```bash
# Procesamiento completo del dataset
python src/data_ingestion/etl.py

# Descarga e ingesta automática (horaria)
python src/data_ingestion/downloader.py
```

### Tareas de VS Code
- **ETL - Procesamiento Completo**: Procesa todo el dataset principal
- **Ingesta de Datos - Horaria**: Descarga datos nuevos cada hora

## 📈 Características Principales

### Sistema de Ingesta
- ✅ **Múltiples fuentes**: OpenAQ API + datos gubernamentales
- ✅ **Procesamiento por chunks**: Manejo eficiente de datasets grandes
- ✅ **Validación automática**: Filtrado de outliers y datos inválidos
- ✅ **Deduplicación**: Prevención de registros duplicados
- ✅ **Fallback automático**: Usa datos locales si API falla

### Base de Datos
- ✅ **Esquema optimizado**: Índices para consultas rápidas
- ✅ **Time series**: Tablas especializadas para análisis temporal
- ✅ **Metadata tracking**: Registro completo de procesos de ingesta
- ✅ **Bulk operations**: Inserción eficiente de grandes volúmenes

### Monitoreo y Logs
- ✅ **Logging estructurado**: Seguimiento detallado de todos los procesos
- ✅ **Métricas de rendimiento**: Tiempos de ejecución y throughput
- ✅ **Reporte de errores**: Identificación y manejo de problemas

## 📊 Estadísticas del Dataset

| Métrica | Valor |
|---------|--------|
| **Total de registros** | 927,999 |
| **Período temporal** | 2010-2024 (14+ años) |
| **Estaciones monitoreadas** | 10 estaciones |
| **Parámetros principales** | PM2.5, PM10, SO2, NO2, O3, CO |
| **Tasa de completitud** | ~85% |
| **Frecuencia de medición** | Horaria |

## 🔍 Calidad de Datos

### Validaciones Implementadas
- **Rangos válidos**: PM2.5 ≥ 0, coordenadas dentro de Lima
- **Detección de outliers**: Método IQR para valores extremos
- **Consistencia temporal**: Validación de timestamps
- **Integridad de estaciones**: Verificación de metadatos

### Limpieza Aplicada
- Eliminación de duplicados por timestamp y estación
- Filtrado de valores negativos o imposibles
- Normalización de nombres de estaciones
- Estandarización de coordenadas geográficas

## 🛠️ Configuración Avanzada

### Variables de Entorno
```bash
# Base de datos
DATABASE_URL=postgresql://user:pass@localhost/lima_air
# o SQLite (por defecto)
DATABASE_URL=sqlite:///lima_air_quality.db

# APIs
OPENAQ_API_KEY=tu_api_key_aqui
```

### Personalización ETL
Editar `config/settings.py` para ajustar:
- Tamaño de chunks de procesamiento
- Umbrales de validación
- Frecuencia de descarga
- Retención de logs

## 📝 Próximos Pasos

### 🚧 Inmediato: Fase 4 - API Backend (En desarrollo activo)

**Objetivo**: Conectar el frontend React con los modelos ML entrenados

#### Tareas pendientes:
- [ ] **FastAPI/Flask setup**: Crear servidor API REST
- [ ] **Endpoint /current**: Mediciones actuales por distrito
- [ ] **Endpoint /forecast**: Predicciones 48h usando modelos ML
- [ ] **Endpoint /risk**: Cálculo de riesgo personalizado para corredores
- [ ] **Documentación Swagger**: API documentation automática
- [ ] **Middleware CORS**: Conexión con frontend React
- [ ] **Autenticación**: API keys para acceso controlado

#### Fórmula de riesgo a implementar:
```python
risk_score = α·pm25_forecast + β·(age/100) + γ·condition_factor
# Mapear a categorías: Bajo/Medio/Alto/Extremo
```

### ⏳ Siguiente: Fase 6 - Sistema de Notificaciones

#### Funcionalidades planificadas:
- [ ] **Integración Twilio**: SMS y WhatsApp
- [ ] **Base de datos suscriptores**: Gestión de usuarios
- [ ] **Sistema de alertas**: Umbrales personalizables
- [ ] **Endpoints de suscripción**: POST /subscribe, /unsubscribe
- [ ] **Scheduler automático**: Envío de alertas horarias
- [ ] **Geolocalización**: Alertas por distrito de interés

### 🎯 Futuro: Fase 7 - Producción

#### Despliegue y escalabilidad:
- [ ] **Contenedorización**: Docker para API y frontend
- [ ] **Cloud deployment**: AWS/Heroku/DigitalOcean
- [ ] **CI/CD pipeline**: Automatización de despliegues
- [ ] **Monitoreo**: Prometheus + Grafana
- [ ] **Dominio personalizado**: HTTPS y SSL
- [ ] **Load balancing**: Para alta disponibilidad

## 🎯 Oportunidades de Monetización

### Modelos de negocio listos para implementar:

1. **SaaS para corredores/deportistas** 💰
   - Base técnica: ✅ 95% completa
   - Falta: API + notificaciones
   - Revenue: $5-15/mes por usuario premium

2. **API B2B para empresas** 💼
   - Modelos ML: ✅ Entrenados y validados
   - Falta: Documentación comercial
   - Revenue: $100-500/mes por empresa

3. **Dashboard white-label** 🏢
   - Frontend: ✅ React profesional
   - Falta: Personalización y multi-tenant
   - Revenue: $1000-5000 implementación

### Planificación Técnica Completada ✅

- ✅ **Análisis exploratorio**: Patrones identificados en `/results/eda/`
- ✅ **Feature engineering**: Variables meteorológicas y temporales
- ✅ **ML Pipeline**: Datasets de entrenamiento optimizados
- ✅ **Validación cruzada**: Modelos evaluados con métricas robustas

## 🤝 Contribución

Este proyecto está en desarrollo activo. Para contribuir:
1. Fork el repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📄 Licencia

[Pendiente definir licencia]

---

**Última actualización**: Agosto 2025  
**Estado actual**: 
- ✅ **Fases 1, 2, 3, 5 completadas** (Ingesta + ML + Dashboard)
- 🚧 **Fase 4 en desarrollo activo** (API REST)
- ⏳ **Fases 6-7 planificadas** (Notificaciones + Despliegue)

**Progreso total**: ~80% completado | **Listo para monetización** con API funcional

**Próxima meta**: API REST funcional conectando React frontend con modelos ML
