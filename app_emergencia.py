# app_emergencia.py (Histórico local + Pronóstico API + Fuente + Gráficos de pronóstico)
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import requests, time, xml.etree.ElementTree as ET

st.set_page_config(page_title="PREDICCIÓN EMERGENCIA AGRÍCOLA - LOLIUM sp.", layout="wide")

# =================== Modelo ANN ===================
class PracticalANNModel:
    def __init__(self, IW, bias_IW, LW, bias_out, low=0.02, medium=0.079):
        self.IW = IW
        self.bias_IW = bias_IW
        self.LW = LW
        self.bias_out = bias_out
        # Orden esperado: [Julian_days, TMAX, TMIN, Prec]
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])
        self.low_thr = low
        self.med_thr = medium

    def tansig(self, x): return np.tanh(x)
    def normalize_input(self, X_real):
        return 2 * (X_real - self.input_min) / (self.input_max - self.input_min) - 1
    def desnormalizar_salida(self, y_norm, ymin=-1, ymax=1):
        return (y_norm - ymin) / (ymax - ymin)
    def _predict_single(self, x_norm):
        z1 = self.IW.T @ x_norm + self.bias_IW
        a1 = self.tansig(z1)
        z2 = self.LW @ a1 + self.bias_out
        return self.tansig(z2)
    def _clasificar(self, valor):
        if valor < self.low_thr: return "Bajo"
        elif valor <= self.med_thr: return "Medio"
        else: return "Alto"
    def predict(self, X_real):
        X_norm = self.normalize_input(X_real)
        emerrel_pred = np.array([self._predict_single(x) for x in X_norm])
        emerrel_desnorm = self.desnormalizar_salida(emerrel_pred)
        emerrel_cumsum = np.cumsum(emerrel_desnorm)
        valor_max_emeac = 8.05
        emer_ac = emerrel_cumsum / valor_max_emeac
        emerrel_diff = np.diff(emer_ac, prepend=0)
        riesgo = np.array([self._clasificar(v) for v in emerrel_diff])
        return pd.DataFrame({
            "EMERREL(0-1)": emerrel_diff,
            "Nivel_Emergencia_relativa": riesgo
        })

# =================== Pronóstico API ===================
API_URL = "https://meteobahia.com.ar/scripts/forecast/for-ta.xml"
API_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://meteobahia.com.ar/",
    "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
}

def _to_float(x):
    try: return float(str(x).replace(",", "."))
    except: return None

@st.cache_data(ttl=15*60, show_spinner=False)
def fetch_forecast(url: str = API_URL, retries: int = 3, backoff: int = 2) -> pd.DataFrame:
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=API_HEADERS, timeout=30)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            days = root.findall(".//forecast/tabular/day")
            rows = []
            for d in days:
                fecha  = d.find("./fecha")
                tmax   = d.find("./tmax")
                tmin   = d.find("./tmin")
                precip = d.find("./precip")
                fval = fecha.get("value") if fecha is not None else None
                if not fval: continue
                rows.append({
                    "Fecha": pd.to_datetime(fval).normalize(),
                    "TMAX": _to_float(tmax.get("value")) if tmax is not None else None,
                    "TMIN": _to_float(tmin.get("value")) if tmin is not None else None,
                    "Prec": _to_float(precip.get("value")) if precip is not None else 0.0,
                })
            if not rows: raise RuntimeError("XML sin días válidos.")
            df = pd.DataFrame(rows).sort_values("Fecha").reset_index(drop=True)
            df["Julian_days"] = df["Fecha"].dt.dayofyear
            return df[["Fecha", "Julian_days", "TMAX", "TMIN", "Prec"]]
        except Exception as e:
            last_err = e
            time.sleep(backoff*(i+1))
    raise RuntimeError(f"No pude obtener el pronóstico desde la API. Último error: {last_err}")

# =================== Utilidades ===================
def validar_columnas(df: pd.DataFrame) -> tuple[bool, str]:
    req = {"Fecha", "Julian_days", "TMAX", "TMIN", "Prec"}
    faltan = req - set(df.columns)
    if faltan: return False, f"Faltan columnas: {', '.join(sorted(faltan))}"
    return True, ""

def obtener_colores(niveles: pd.Series):
    m = niveles.map({"Bajo": "green", "Medio": "orange", "Alto": "red"})
    return m.fillna("gray")

def detectar_fuera_rango(X_real: np.ndarray, input_min: np.ndarray, input_max: np.ndarray) -> bool:
    out = (X_real < input_min) | (X_real > input_max)
    return bool(np.any(out))

@st.cache_data(show_spinner=False)
def load_weights(base_dir: Path):
    IW = np.load(base_dir / "IW.npy")
    bias_IW = np.load(base_dir / "bias_IW.npy")
    LW = np.load(base_dir / "LW.npy")
    bias_out = np.load(base_dir / "bias_out.npy")
    return IW, bias_IW, LW, bias_out

# =================== UI ===================
st.title("PREDICCIÓN EMERGENCIA AGRÍCOLA - LOLIUM sp. (Histórico + Pronóstico API)")

st.sidebar.header("Fuente de datos")
fuente = st.sidebar.radio(
    "Elegí la fuente",
    ["Histórico local + Pronóstico (API)", "Subir histórico + usar Pronóstico (API)"]
)

st.sidebar.header("Configuración")
umbral_usuario = st.sidebar.number_input("Umbral de EMEAC para 100%", min_value=1.2, max_value=3.0, value=2.70, step=0.01, format="%.2f")

st.sidebar.header("Validaciones")
mostrar_fuera_rango = st.sidebar.checkbox("Avisar datos fuera de rango de entrenamiento", value=False)

if st.sidebar.button("Forzar recarga de datos"):
    st.cache_data.clear()

# Pesos modelo
try:
    base = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    IW, bias_IW, LW, bias_out = load_weights(base)
except FileNotFoundError as e:
    st.error("Error al cargar archivos del modelo (IW.npy, bias_IW.npy, LW.npy, bias_out.npy). "
             f"Ruta buscada: {base}. Detalle: {e}")
    st.stop()

modelo = PracticalANNModel(IW, bias_IW, LW, bias_out)

# =================== Cargar histórico ===================
df_hist = None
hist_path_default = Path("meteo_daily.csv")

if fuente == "Histórico local + Pronóstico (API)":
    if hist_path_default.exists():
        try:
            df_hist = pd.read_csv(hist_path_default, parse_dates=["Fecha"])
        except Exception as e:
            st.error(f"No pude leer el histórico local en {hist_path_default}: {e}")
    else:
        st.warning(f"No se encontró {hist_path_default}. Podés subir el histórico en la opción: 'Subir histórico + usar Pronóstico (API)'.")

elif fuente == "Subir histórico + usar Pronóstico (API)":
    up = st.file_uploader("Subí el histórico (.csv) con columnas: Fecha, Julian_days, TMAX, TMIN, Prec", type=["csv"])
    if up is not None:
        try:
            df_hist = pd.read_csv(up, parse_dates=["Fecha"])
        except Exception as e:
            st.error(f"No pude leer el CSV subido: {e}")

# Validar/limpiar histórico
if df_hist is not None:
    ok, msg = validar_columnas(df_hist)
    if not ok:
        st.error(f"Histórico inválido: {msg}")
        df_hist = None
    else:
        cols_num = ["Julian_days", "TMAX", "TMIN", "Prec"]
        df_hist[cols_num] = df_hist[cols_num].apply(pd.to_numeric, errors="coerce")
        df_hist = df_hist.dropna(subset=cols_num).copy()
        df_hist["Fecha"] = pd.to_datetime(df_hist["Fecha"]).dt.normalize()
        df_hist["Julian_days"] = df_hist["Fecha"].dt.dayofyear
        df_hist = df_hist.sort_values("Fecha").reset_index(drop=True)

# =================== Pronóstico (API) ===================
df_fcst = None
try:
    df_fcst = fetch_forecast()
    if df_fcst is not None and not df_fcst.empty:
        st.success(f"Pronóstico API cargado: {df_fcst['Fecha'].min().date()} → {df_fcst['Fecha'].max().date()} · {len(df_fcst)} días")
except Exception as e:
    st.error(f"Fallo al obtener pronóstico desde API: {e}")

# =================== Gráficos/tabla del PRONÓSTICO (API) ===================
if df_fcst is not None and not df_fcst.empty:
    st.subheader("📊 Pronóstico Meteorológico - API MeteoBahía (Tres Arroyos)")
    # Temperaturas
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=df_fcst["Fecha"], y=df_fcst["TMAX"], mode="lines+markers",
        name="TMAX (°C)", hovertemplate="Fecha: %{x|%d-%b-%Y}<br>TMAX: %{y:.1f} °C"
    ))
    fig_temp.add_trace(go.Scatter(
        x=df_fcst["Fecha"], y=df_fcst["TMIN"], mode="lines+markers",
        name="TMIN (°C)", hovertemplate="Fecha: %{x|%d-%b-%Y}<br>TMIN: %{y:.1f} °C"
    ))
    fig_temp.update_layout(yaxis_title="Temperatura (°C)", xaxis_title="Fecha", hovermode="x unified", height=380)
    st.plotly_chart(fig_temp, use_container_width=True, theme="streamlit")

    # Precipitación
    fig_prec = go.Figure()
    fig_prec.add_trace(go.Bar(
        x=df_fcst["Fecha"], y=df_fcst["Prec"], name="Precipitación (mm)",
        hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Prec: %{y:.1f} mm"
    ))
    fig_prec.update_layout(yaxis_title="Precipitación (mm)", xaxis_title="Fecha", hovermode="x unified", height=360)
    st.plotly_chart(fig_prec, use_container_width=True, theme="streamlit")

    # Tabla + descarga
    st.dataframe(df_fcst[["Fecha", "TMAX", "TMIN", "Prec"]], use_container_width=True)
    st.download_button(
        "Descargar pronóstico API",
        df_fcst.to_csv(index=False).encode("utf-8"),
        "pronostico_api.csv",
        "text/csv"
    )

st.divider()

# =================== Combinar (Histórico < hoy) + (API >= hoy) ===================
dfs = []
if df_hist is not None and df_fcst is not None:
    today = pd.Timestamp.today().normalize()
    df_hist_past = df_hist[df_hist["Fecha"] < today].copy()
    df_fcst_today_fwd = df_fcst[df_fcst["Fecha"] >= today].copy()
    # Etiquetas de fuente
    df_hist_past["Fuente"] = "Histórico CSV"
    df_fcst_today_fwd["Fuente"] = "API MeteoBahía"
    df_all = pd.concat([df_hist_past, df_fcst_today_fwd], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["Fecha"], keep="last").sort_values("Fecha").reset_index(drop=True)
    df_all["Julian_days"] = df_all["Fecha"].dt.dayofyear
    # Resumen de fuente
    vc = df_all["Fuente"].value_counts().to_dict()
    st.caption(f"Fuente de datos combinada → Histórico: {vc.get('Histórico CSV', 0)} · API: {vc.get('API MeteoBahía', 0)}")
    dfs.append(("Histórico+Pronóstico", df_all))
elif df_fcst is not None and df_hist is None:
    st.info("Usando solo Pronóstico (API) porque no hay histórico válido disponible.")
    df_fcst["Fuente"] = "API MeteoBahía"
    dfs.append(("Solo_Pronóstico", df_fcst))
elif df_hist is not None and df_fcst is None:
    st.info("Usando solo Histórico porque falló el pronóstico de la API.")
    df_hist["Fuente"] = "Histórico CSV"
    dfs.append(("Solo_Histórico", df_hist))
else:
    st.stop()

# =================== Procesamiento y gráficos (modelo) ===================
def plot_and_table(nombre, df):
    df = df.sort_values("Fecha").reset_index(drop=True)
    X_real = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(dtype=float)
    fechas = pd.to_datetime(df["Fecha"])

    if mostrar_fuera_rango and detectar_fuera_rango(X_real, modelo.input_min, modelo.input_max):
        st.info(f"⚠️ {nombre}: hay valores fuera del rango de entrenamiento ({modelo.input_min} a {modelo.input_max}).")

    pred = modelo.predict(X_real)
    pred["Fecha"] = fechas
    pred["Julian_days"] = df["Julian_days"]
    if "Fuente" in df.columns:
        pred["Fuente"] = df["Fuente"].values

    pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()
    pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()

    pred["EMEAC (0-1) - mínimo"] = pred["EMERREL acumulado"] / 1.2
    pred["EMEAC (0-1) - máximo"] = pred["EMERREL acumulado"] / 3.0
    pred["EMEAC (0-1) - ajustable"] = pred["EMERREL acumulado"] / umbral_usuario
    pred["EMEAC (%) - mínimo"] = pred["EMEAC (0-1) - mínimo"] * 100
    pred["EMEAC (%) - máximo"] = pred["EMEAC (0-1) - máximo"] * 100
    pred["EMEAC (%) - ajustable"] = pred["EMEAC (0-1) - ajustable"] * 100

    years = pred["Fecha"].dt.year.unique()
    yr = int(years[0]) if len(years) == 1 else int(st.sidebar.selectbox("Año a mostrar (reinicio 1/feb → 1/sep)", sorted(years), key=f"year_select_{nombre}"))

    fecha_inicio_rango = pd.Timestamp(year=yr, month=2, day=1)
    fecha_fin_rango    = pd.Timestamp(year=yr, month=9, day=1)
    mask = (pred["Fecha"] >= fecha_inicio_rango) & (pred["Fecha"] <= fecha_fin_rango)
    pred_vis = pred.loc[mask].copy()
    if pred_vis.empty:
        st.warning(f"No hay datos entre {fecha_inicio_rango.date()} y {fecha_fin_rango.date()} para {nombre}.")
        return

    pred_vis["EMERREL acumulado (reiniciado)"] = pred_vis["EMERREL(0-1)"].cumsum()
    pred_vis["EMEAC (0-1) - mínimo (rango)"]    = pred_vis["EMERREL acumulado (reiniciado)"] / 1.2
    pred_vis["EMEAC (0-1) - máximo (rango)"]    = pred_vis["EMERREL acumulado (reiniciado)"] / 3.0
    pred_vis["EMEAC (0-1) - ajustable (rango)"] = pred_vis["EMERREL acumulado (reiniciado)"] / umbral_usuario
    pred_vis["EMEAC (%) - mínimo (rango)"]      = pred_vis["EMEAC (0-1) - mínimo (rango)"] * 100
    pred_vis["EMEAC (%) - máximo (rango)"]      = pred_vis["EMEAC (0-1) - máximo (rango)"] * 100
    pred_vis["EMEAC (%) - ajustable (rango)"]   = pred_vis["EMEAC (0-1) - ajustable (rango)"] * 100

    pred_vis["EMERREL_MA5_rango"] = pred_vis["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()
    colores_vis = obtener_colores(pred_vis["Nivel_Emergencia_relativa"])

    st.subheader("EMERGENCIA RELATIVA DIARIA - BORDENAVE")
    fig_er = go.Figure()
    fig_er.add_bar(
        x=pred_vis["Fecha"], y=pred_vis["EMERREL(0-1)"],
        marker=dict(color=colores_vis.tolist()),
        hovertemplate=("Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.3f}<br>Nivel: %{customdata}"),
        customdata=pred_vis["Nivel_Emergencia_relativa"], name="EMERREL (0-1)",
    )
    fig_er.add_trace(go.Scatter(
        x=pred_vis["Fecha"], y=pred_vis["EMERREL_MA5_rango"],
        mode="lines", name="Media móvil 5 días (rango)",
        hovertemplate="Fecha: %{x|%d-%b-%Y}<br>MA5: %{y:.3f}<extra></extra>"
    ))
    fig_er.add_trace(go.Scatter(
        x=pred_vis["Fecha"], y=pred_vis["EMERREL_MA5_rango"],
        mode="lines", line=dict(width=0), fill="tozeroy",
        fillcolor="rgba(135, 206, 250, 0.3)", name="Área MA5",
        hoverinfo="skip", showlegend=False
    ))
    low_thr = float(modelo.low_thr); med_thr = float(modelo.med_thr)
    fig_er.add_trace(go.Scatter(x=[fecha_inicio_rango, fecha_fin_rango], y=[low_thr, low_thr],
        mode="lines", line=dict(color="green", dash="dot"),
        name=f"Bajo (≤ {low_thr:.3f})", hoverinfo="skip"))
    fig_er.add_trace(go.Scatter(x=[fecha_inicio_rango, fecha_fin_rango], y=[med_thr, med_thr],
        mode="lines", line=dict(color="orange", dash="dot"),
        name=f"Medio (≤ {med_thr:.3f})", hoverinfo="skip"))
    fig_er.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
        line=dict(color="red", dash="dot"), name=f"Alto (> {med_thr:.3f})",
        hoverinfo="skip", showlegend=True))
    fig_er.update_layout(xaxis_title="Fecha", yaxis_title="EMERREL (0-1)",
                         hovermode="x unified", legend_title="Referencias", height=650)
    fig_er.update_xaxes(range=[fecha_inicio_rango, fecha_fin_rango], dtick="M1", tickformat="%b")
    fig_er.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig_er, use_container_width=True, theme="streamlit")

    st.subheader("EMERGENCIA ACUMULADA DIARIA - BORDENAVE")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - máximo (rango)"],
                             mode="lines", line=dict(width=0), name="Máximo (reiniciado)",
                             hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - mínimo (rango)"],
                             mode="lines", line=dict(width=0), fill="tonexty", name="Mínimo (reiniciado)",
                             hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - ajustable (rango)"],
                             mode="lines", name="Umbral ajustable (reiniciado)",
                             hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Ajustable: %{y:.1f}%<extra></extra>",
                             line=dict(width=2.5)))
    fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - mínimo (rango)"],
                             mode="lines", name="Umbral mínimo (reiniciado)",
                             line=dict(dash="dash", width=1.5),
                             hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - máximo (rango)"],
                             mode="lines", name="Umbral máximo (reiniciado)",
                             line=dict(dash="dash", width=1.5),
                             hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"))
    for nivel in [25, 50, 75, 90]:
        fig.add_hline(y=nivel, line_dash="dash", opacity=0.6, annotation_text=f"{nivel}%")
    fig.update_layout(xaxis_title="Fecha", yaxis_title="EMEAC (%)", yaxis=dict(range=[0, 100]),
                      hovermode="x unified", legend_title="Referencias", height=600)
    fig.update_xaxes(range=[fecha_inicio_rango, fecha_fin_rango], dtick="M1", tickformat="%b")
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    st.subheader(f"Resultados (1/feb → 1/sep) - {nombre}")
    col_emeac = "EMEAC (%) - ajustable (rango)" if "EMEAC (%) - ajustable (rango)" in pred_vis.columns else "EMEAC (%) - ajustable"
    cols_tabla = ["Fecha", "Julian_days", "Nivel_Emergencia_relativa", col_emeac]
    if "Fuente" in pred_vis.columns:
        cols_tabla.insert(2, "Fuente")
    tabla = pred_vis[cols_tabla].rename(
        columns={"Nivel_Emergencia_relativa": "Nivel de EMERREL", col_emeac: "EMEAC (%)"}
    )
    st.dataframe(tabla, use_container_width=True)
    csv = tabla.to_csv(index=False).encode("utf-8")
    st.download_button(f"Descargar resultados (rango) - {nombre}", csv, f"{nombre}_resultados_rango.csv", "text/csv")

for nombre, df in dfs:
    plot_and_table(nombre, df)
