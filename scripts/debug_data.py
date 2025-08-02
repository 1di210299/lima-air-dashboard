#!/usr/bin/env python3
"""
Debug script para verificar datos limpios
"""

import pandas as pd
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    print("🔍 DEPURANDO DATOS LIMPIOS")
    print("="*50)
    
    # Cargar datos limpios
    df = pd.read_csv("data/lima_air_quality_cleaned.csv")
    print(f"📊 Registros cargados: {len(df):,}")
    print(f"📋 Columnas: {list(df.columns)}")
    
    # Verificar timestamp
    print(f"\n🕒 COLUMNA TIMESTAMP:")
    print(f"  • Tipo: {df['timestamp'].dtype}")
    print(f"  • Primeros 5 valores:")
    for i, val in enumerate(df['timestamp'].head()):
        print(f"    {i+1}. {val} (tipo: {type(val)})")
    
    # Verificar valores únicos de timestamp
    print(f"  • Valores únicos de timestamp: {df['timestamp'].nunique()}")
    print(f"  • Valores nulos: {df['timestamp'].isnull().sum()}")
    
    # Intentar convertir timestamp
    print(f"\n🔧 INTENTANDO CONVERSIÓN:")
    try:
        df['timestamp_converted'] = pd.to_datetime(df['timestamp'], errors='coerce')
        print(f"  ✅ Conversión exitosa")
        print(f"  • Valores nulos después de conversión: {df['timestamp_converted'].isnull().sum()}")
        print(f"  • Tipo después de conversión: {df['timestamp_converted'].dtype}")
        print(f"  • Primeros 3 valores convertidos:")
        for i, val in enumerate(df['timestamp_converted'].head(3)):
            print(f"    {i+1}. {val}")
            
    except Exception as e:
        print(f"  ❌ Error en conversión: {str(e)}")
    
    # Verificar estaciones
    print(f"\n🏭 ESTACIONES:")
    print(f"  • Estaciones únicas: {df['station_name'].nunique()}")
    print(f"  • Estaciones: {df['station_name'].unique()}")
    
    # Verificar contaminantes
    pollutants = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
    available = [col for col in pollutants if col in df.columns]
    print(f"\n💨 CONTAMINANTES DISPONIBLES: {available}")
    
    for col in available:
        non_null = df[col].notna().sum()
        print(f"  • {col}: {non_null:,} valores no nulos ({non_null/len(df)*100:.1f}%)")

if __name__ == "__main__":
    main()
