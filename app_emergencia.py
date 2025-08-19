#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import time
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

API_DEFAULT = "https://meteobahia.com.ar/scripts/forecast/for-ta.xml"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://meteobahia.com.ar/",
    "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
}

def to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return None

def fetch_xml(url: str, timeout: int, retries: int, backoff: int) -> bytes:
    last_err = None
    for i in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            last_err = e
            if i < retries - 1:
                time.sleep(backoff * (i + 1))
    raise RuntimeError(f"No se pudo obtener la API ({url}). Último error: {last_err}")

def parse_forecast_xml(xml_bytes: bytes) -> pd.DataFrame:
    """
    Espera estructura:
      <forecast>
        <tabular>
          <day>
            <fecha value="YYYY-MM-DD" />
            <tmax  value=".." />
            <tmin  value=".." />
            <precip value=".." />
          </day>...
        </tabular>
      </forecast>
    """
    root = ET.fromstring(xml_bytes)
    days = root.findall(".//forecast/tabular/day")
    rows = []
    for d in days:
        fecha  = d.find("./fecha")
        tmax   = d.find("./tmax")
        tmin   = d.find("./tmin")
        precip = d.find("./precip")

        fval = fecha.get("value") if fecha is not None else None
        if not fval:
            # sin fecha => día inválido
            continue

        rows.append({
            "Fecha": pd.to_datetime(fval).normalize(),
            "TMAX": to_float(tmax.get("value")) if tmax is not None else None,
            "TMIN": to_float(tmin.get("value")) if tmin is not None else None,
            "Prec": to_float(precip.get("value")) if precip is not None else 0.0,
        })

    if not rows:
        raise ValueError("XML sin <day> válidos.")

    df = pd.DataFrame(rows).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    # Reordenar columnas y devolver
    return df[["Fecha", "Julian_days", "TMAX", "TMIN", "Prec"]]

def validate_dataframe(
    df: pd.DataFrame,
    require_today: bool,
    require_future: bool,
    min_days: int,
) -> dict:
    # Estructura mínima
    required = {"Fecha", "Julian_days", "TMAX", "TMIN", "Prec"}
    missing_cols = sorted(list(required - set(df.columns)))
    if missing_cols:
        return {"ok": False, "reason": f"Faltan columnas: {', '.join(missing_cols)}"}

    # Tipos y nulos
    numeric_cols = ["TMAX", "TMIN", "Prec"]
    nulls = df[numeric_cols].isna().sum().to_dict()

    # Métricas temporales
    today = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None).normalize()
    has_today = bool((df["Fecha"] == today).any())
    has_future = bool((df["Fecha"] > today).any())

    # Tamaño
    n_days = int(len(df))

    ok = True
    problems = []

    if min_days and n_days < min_days:
        ok = False
        problems.append(f"Se esperaban al menos {min_days} días; vinieron {n_days}.")

    if require_today and not has_today:
        ok = False
        problems.append("El pronóstico NO incluye el día de hoy.")

    if require_future and not has_future:
        ok = False
        problems.append("El pronóstico NO incluye días futuros.")

    # Criterio: nulos tolerados? Depende del uso. Por defecto, exigimos 0.
    for c in numeric_cols:
        if nulls[c] > 0:
            ok = False
            problems.append(f"Columna {c} tiene {nulls[c]} valores nulos/no numéricos.")

    return {
        "ok": ok,
        "problems": problems,
        "summary": {
            "rango_fechas": f"{df['Fecha'].min().date()} → {df['Fecha'].max().date()}",
            "n_dias": n_days,
            "incluye_hoy": has_today,
            "incluye_futuro": has_future,
            "n_nulls": nulls,
        },
    }

def main():
    ap = argparse.ArgumentParser(
        description="Verifica que la API de MeteoBahía (forecast) devuelve datos completos y válidos."
    )
    ap.add_argument("--url", default=API_DEFAULT, help="URL de la API XML.")
    ap.add_argument("--timeout", type=int, default=30, help="Timeout de request (s).")
    ap.add_argument("--retries", type=int, default=3, help="Reintentos ante error.")
    ap.add_argument("--backoff", type=int, default=2, help="Backoff lineal entre reintentos (s * intento).")
    ap.add_argument("--min-days", type=int, default=3, help="Mínimo de días esperados en el pronóstico.")
    ap.add_argument("--require-today", action="store_true", help="Exigir que incluya el día de hoy.")
    ap.add_argument("--require-future", action="store_true", help="Exigir que incluya al menos 1 día futuro.")
    ap.add_argument("--out", default="", help="Ruta para guardar CSV con el pronóstico validado (opcional).")
    args = ap.parse_args()

    try:
        xml = fetch_xml(args.url, args.timeout, args.retries, args.backoff)
    except Exception as e:
        print(f"[ERROR] Fetch falló: {e}", file=sys.stderr)
        return 2

    # Parseo
    try:
        df = parse_forecast_xml(xml)
    except Exception as e:
        print(f"[ERROR] XML inválido o estructura inesperada: {e}", file=sys.stderr)
        return 3

    # Validación
    result = validate_dataframe(
        df,
        require_today=args.require_today,
        require_future=args.require_future,
        min_days=args.min_days,
    )

    # Resumen
    print("=== Diagnóstico API MeteoBahía (forecast) ===")
    print(f"URL: {args.url}")
    print(f"Rango fechas: {result['summary']['rango_fechas']}")
    print(f"Días recibidos: {result['summary']['n_dias']}")
    print(f"Incluye hoy: {result['summary']['incluye_hoy']}")
    print(f"Incluye futuro: {result['summary']['incluye_futuro']}")
    print(f"Nulos (TMAX/TMIN/Prec): {result['summary']['n_nulls']}")
    print()

    if not result["ok"]:
        print("[FALLO] La validación no pasó por:")
        for p in result["problems"]:
            print(f" - {p}")
        # Guardar CSV aunque falle puede ayudar al debug
        if args.out:
            try:
                df.to_csv(args.out, index=False)
                print(f"(Se guardó CSV para inspección en: {args.out})")
            except Exception as e:
                print(f"(No se pudo guardar CSV: {e})")
        return 4

    print("[OK] La API devuelve los datos necesarios y válidos para el modelo.")
    if args.out:
        try:
            df.to_csv(args.out, index=False)
            print(f"CSV guardado en: {args.out}")
        except Exception as e:
            print(f"No se pudo guardar CSV: {e}")
            return 5

    return 0

if __name__ == "__main__":
    sys.exit(main())
