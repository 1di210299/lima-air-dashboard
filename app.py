# app.py

import os
import requests
import pandas as pd
from dotenv import load_dotenv, find_dotenv

# --------------------------------------------------
# 1. Carga de la API Key desde .env
# --------------------------------------------------
load_dotenv(find_dotenv())
API_KEY = os.getenv("OPENAQ_API_KEY")
if not API_KEY:
    raise RuntimeError("Define OPENAQ_API_KEY en tu archivo .env")

HEADERS = {
    "Accept": "application/json",
    "X-API-Key": API_KEY
}

# Bounding box aproximado de Lima Metropolitana
LIMA_BBOX = {
    "lat_min": -12.30, "lat_max": -11.80,
    "lon_min": -77.30, "lon_max": -76.90
}

# --------------------------------------------------
# 2. Función: obtener lecturas actuales (Latest)
# --------------------------------------------------
def fetch_current_pm25(limit: int = 1000) -> pd.DataFrame:
    """
    Obtiene las lecturas más recientes de PM2.5 en Lima usando
    GET /v3/parameters/2/latest.
    """
    url = "https://api.openaq.org/v3/parameters/2/latest"
    resp = requests.get(url, headers=HEADERS, params={"limit": limit})
    resp.raise_for_status()
    data = resp.json().get("results", [])

    records = []
    for e in data:
        coords = e.get("coordinates") or {}
        lat, lon = coords.get("latitude"), coords.get("longitude")
        # Filtrar solo estaciones dentro de Lima
        if None in (lat, lon):
            continue
        if not (LIMA_BBOX["lat_min"] <= lat <= LIMA_BBOX["lat_max"] and
                LIMA_BBOX["lon_min"] <= lon <= LIMA_BBOX["lon_max"]):
            continue

        # Extraer nombre de estación (opcionalmente via lookup)
        loc_id = e.get("locationsId")
        name = str(loc_id)
        if loc_id:
            loc_resp = requests.get(f"https://api.openaq.org/v3/locations/{loc_id}",
                                    headers=HEADERS)
            if loc_resp.ok:
                locs = loc_resp.json().get("results", [])
                if locs:
                    name = locs[0].get("name", name)

        records.append({
            "station_id":   loc_id,
            "station_name": name,
            "latitude":     lat,
            "longitude":    lon,
            "value":        e.get("value"),
            "unit":         e.get("unit"),
            "datetime":     e.get("datetime", {}).get("utc")
        })

    return pd.DataFrame(records)


# --------------------------------------------------
# 3. Función: obtener histórico para un rango dado
# --------------------------------------------------
def fetch_historical_pm25(date_from: str, date_to: str,
                          limit: int = 10000) -> pd.DataFrame:
    """
    Obtiene historial de PM2.5 en Lima en el rango de fechas indicado usando
    GET /v3/measurements.
    - date_from, date_to: 'YYYY-MM-DD' o 'YYYY-MM-DDTHH:MM:SSZ'
    """
    url = "https://api.openaq.org/v3/measurements"
    # parámetros: ciudad, parámetro, rango de fechas, orden descendente
    params = {
        "city":       "Lima",
        "parameter":  "pm25",
        "date_from":  date_from,
        "date_to":    date_to,
        "limit":      limit,
        "order_by":   "datetime",
        "sort":       "desc"
    }
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    data = resp.json().get("results", [])

    records = []
    for e in data:
        loc   = e.get("location")
        coords= e.get("coordinates") or {}
        lat, lon = coords.get("latitude"), coords.get("longitude")
        # Opcional: filtrar fuera de Lima si fuese necesario
        if None in (lat, lon):
            continue
        records.append({
            "station_name": loc,
            "latitude":     lat,
            "longitude":    lon,
            "value":        e.get("value"),
            "unit":         e.get("unit"),
            "datetime":     e.get("date", {}).get("utc")
        })

    return pd.DataFrame(records)


# --------------------------------------------------
# 4. Ejecución principal
# --------------------------------------------------
def main():
    # Datos actuales
    print("🔄 Obteniendo lecturas actuales de PM2.5 en Lima…")
    df_current = fetch_current_pm25(limit=500)
    print(f"  → {len(df_current)} estaciones encontradas.\n")
    print(df_current.head(), "\n")

    # Datos históricos de 2025
    print("📅 Obteniendo histórico de PM2.5 en 2025 (01-01 al 08-02)…")
    df_hist_2025 = fetch_historical_pm25("2025-01-01", "2025-08-02", limit=10000)
    print(f"  → {len(df_hist_2025)} lecturas obtenidas.\n")
    print(df_hist_2025.head(), "\n")

    # (Opcional) Guardar a CSV
    df_current.to_csv("current_pm25_lima.csv", index=False)
    df_hist_2025.to_csv("historical_pm25_2025_lima.csv", index=False)
    print("✅ Archivos CSV generados: current_pm25_lima.csv, historical_pm25_2025_lima.csv")

if __name__ == "__main__":
    main()
