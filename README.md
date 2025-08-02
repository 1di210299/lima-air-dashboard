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

### 🚀 Siguientes Fases:
2. **Preprocesamiento y limpieza**
3. **Modelado y predicción ML**
4. **API Backend**
5. **Dashboard Web**
6. **Sistema de notificaciones**
7. **Despliegue y producción**

## 🏗️ Arquitectura del Sistema

```
lima-air-dashboard/
├── src/
│   ├── database/          # Modelos y conexiones de BD
│   │   ├── models.py      # Esquemas SQLAlchemy
│   │   └── connection.py  # Gestión de conexiones
│   └── data_ingestion/    # Pipeline de ingesta
│       ├── etl.py         # Procesamiento ETL
│       └── downloader.py  # Descarga automática
├── config/
│   └── settings.py        # Configuración del sistema
├── data/
│   └── lima_air_quality_complete.csv  # Dataset principal
├── logs/                  # Logs del sistema
└── requirements.txt       # Dependencias
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

### Fase 2: Preprocesamiento (En desarrollo)
- [ ] Pipeline de limpieza avanzada
- [ ] Imputación de valores faltantes
- [ ] Agregaciones temporales (diaria, semanal, mensual)
- [ ] Features derivadas (estacionalidad, tendencias)
- [ ] Normalización para ML

### Planificación Técnica
1. **Análisis exploratorio**: Patrones estacionales y tendencias
2. **Feature engineering**: Variables meteorológicas y temporales
3. **Preparación para ML**: Datasets de entrenamiento y validación

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
**Estado**: Fase 1 completada ✅ | Fase 2 en desarrollo 🚧
