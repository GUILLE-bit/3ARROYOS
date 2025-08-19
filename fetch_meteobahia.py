import time, requests, pandas as pd, numpy as np, xml.etree.ElementTree as ET
from pathlib import Path

# NUEVA URL (Tres Arroyos)
URL_FCST = "https://meteobahia.com.ar/scripts/forecast/for-ta.xml"
OUT = Path("LOLIUMTRES/blob/gh-pages/meteo_daily.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://meteobahia.com.ar/",
    "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
}

def fetch_fcst_xml(url=URL_FCST, timeout=30, retries=3, backoff=2):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last = e
            time.sleep(backoff*(i+1))
    raise RuntimeError(f"Fetch forecast failed: {last}")

def parse_fcst(xml_bytes: bytes) -> pd.DataFrame:
    root = ET.fromstring(xml_bytes)
    days = root.findall(".//forecast/tabular/day")
    rows = []

    def to_f(x):
        try:
            return float(str(x).replace(",", "."))
        except:
            return None

    for d in days:
        fecha  = d.find("./fecha")
        tmax   = d.find("./tmax")
        tmin   = d.find("./tmin")
        precip = d.find("./precip")

        fval = fecha.get("value") if fecha is not None else None
        if not fval:
            continue

        rows.append({
            "Fecha": pd.to_datetime(fval).normalize(),
            "TMAX": to_f(tmax.get("value")) if tmax is not None else None,
            "TMIN": to_f(tmin.get("value")) if tmin is not None else None,
            "Prec": to_f(precip.get("value")) if precip is not None else 0.0,
        })

    if not rows:
        raise RuntimeError("Forecast XML sin <day> válidos.")

    df = pd.DataFrame(rows).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    return df[["Fecha", "Julian_days", "TMAX", "TMIN", "Prec"]]

# --- Mismo comportamiento para unir con histórico ---
def main():
    xmlb = fetch_fcst_xml()
    df_fcst = parse_fcst(xmlb)

    df_fcst.to_csv(OUT, index=False)
    print(f"[OK] Guardado {OUT} con {len(df_fcst)} filas | {df_fcst['Fecha'].min().date()} → {df_fcst['Fecha'].max().date()}")

if __name__ == "__main__":
    main()
