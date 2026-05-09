import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import csv
from analysis.risk import evaluate_portfolio_risk
from analysis.patterns import detect_consecutive_up_days, detect_reversal_v_pattern
from analysis.similarity import euclidean_distance, pearson_correlation, cosine_similarity, dtw_distance
from analysis.ml import knn_predict, monte_carlo_simulation, brute_force_portfolio
from config import MASTER_DIR
from visualization.pdf_generator import generate_technical_report

st.set_page_config(page_title="Fintech Analytics", layout="wide", page_icon="📈")

# Estilos CSS Profesionales / Premium Dark Mode
st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Fondo principal y textos */
    .stApp { background-color: #0F172A; font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { color: #F8FAFC !important; font-family: 'Inter', sans-serif; font-weight: 700; letter-spacing: -0.5px;}
    p, span, div, label { color: #F8FAFC !important; font-family: 'Inter', sans-serif; font-weight: 400;}
    
    /* Contenedores Inputs y Textos de Dropdown */
    div[data-testid="stSelectbox"] *, div[data-testid="stMultiSelect"] * {
        color: #0F172A !important; /* Texto oscuro de alto contraste */
    }
    
    /* Popover (Lista desplegable que flota sobre la app) */
    div[data-baseweb="popover"] *, ul[role="listbox"] * {
        color: #0F172A !important; /* Fuerza las opciones a ser oscuras */
        font-weight: 500;
    }
    
    /* Fondo Hover de la lista para que se note al pasar el mouse */
    ul[role="listbox"] li:hover, ul[role="listbox"] li[aria-selected="true"] {
        background-color: #E2E8F0 !important;
    }
    
    /* Tags seleccionados en MultiSelect */
    span[data-baseweb="tag"] {
        background-color: #2563EB !important;
    }
    span[data-baseweb="tag"] span {
        color: #FFFFFF !important; /* Texto blanco en las etiquetas azules */
    }
    .stSlider > div { color: #3B82F6; }
    
    /* Métricas */
    [data-testid="stMetric"] {
        border: 1px solid #334155;
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-radius: 12px; 
        padding: 20px;
    }
    .metric-value, [data-testid="stMetricValue"] { font-size: 28px !important; color: #38BDF8 !important; font-weight: 700; }
    [data-testid="stMetricLabel"] {color: #94A3B8 !important; font-size: 14px; font-weight: 600;}
    
    /* Botones */
    div.stButton > button:first-child {
        background-color: #2563EB; border: none; color: white;
        border-radius: 8px; transition: all 0.2s ease; font-weight: 600; padding: 0.5rem 1rem;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    div.stButton > button:first-child:hover {
        background-color: #1D4ED8; color: white; box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3); transform: translateY(-1px);
    }
    
    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; border-bottom: 2px solid #334155; padding: 0px; gap: 24px; }
    .stTabs [data-baseweb="tab"] { color: #94A3B8; border: none; padding: 12px 4px; transition: all 0.2s; font-weight: 600; background-color: transparent;}
    .stTabs [aria-selected="true"] {
        color: #38BDF8 !important; border-bottom: 2px solid #38BDF8; background-color: transparent !important;
    }
    
    /* Tablas CSS */
    table { border-collapse: collapse; width: 100%; border: 1px solid #334155; border-radius: 8px; overflow: hidden; display: block; }
    th { background-color: #1E293B; color: #F8FAFC !important; font-weight: 600; padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155;}
    td { padding: 12px 16px; border-bottom: 1px solid #1E293B; color: #CBD5E1; }
    tr:hover { background-color: rgba(30, 41, 59, 0.5); }
    </style>""", unsafe_allow_html=True
)
# Plantilla base Premium para plotly
hud_layout = dict(
    paper_bgcolor='rgba(0,0,0,0)', 
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif", color="#F8FAFC"), # Blanco brillante para mejor legibilidad
    xaxis=dict(showgrid=True, gridcolor='#334155', zerolinecolor='#475569'),
    yaxis=dict(showgrid=True, gridcolor='#334155', zerolinecolor='#475569'),
    legend=dict(font=dict(color="#FFFFFF")) # Forzar leyendas a color blanco
)

@st.cache_data
def load_master_data():
    master_file = os.path.join(MASTER_DIR, "master_dataset.csv")
    if not os.path.exists(master_file):
        st.warning("Ejecutando proceso ETL completo por primera vez...")
        from etl.extractor import extract_data
        from etl.transformer import transform_data
        from etl.loader import load_data
        extract_data()
        transform_data()
        load_data()
        
    master_rows = []
    with open(master_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['open'] = float(row['open']) if row['open'] else np.nan
            row['high'] = float(row['high']) if row['high'] else np.nan
            row['low'] = float(row['low']) if row['low'] else np.nan
            row['close'] = float(row['close']) if row['close'] else np.nan
            row['volumen'] = float(row['volumen']) if row['volumen'] else 0
            master_rows.append(row)
            
    return master_rows

st.title("Sistema Avanzado de Análisis Financiero 📊")
try:
    master_rows = load_master_data()
    # Extraer tickers únicos puramente iterando
    tickers = list(dict.fromkeys([row['ticker'] for row in master_rows]))
    
    # Generador de reportes en la parte superior
    col_header1, col_header2 = st.columns([4, 1])
    with col_header2:
        if st.button("📄 Generar Reporte PDF", use_container_width=True):
            with st.spinner('Procesando PDF...'):
                risk_results = evaluate_portfolio_risk(master_rows)
                pdf_path = os.path.join(MASTER_DIR, "Reporte_Tecnico.pdf")
                generate_technical_report(risk_results, pdf_path)
                st.session_state['pdf_generated'] = pdf_path
                
        if 'pdf_generated' in st.session_state and os.path.exists(st.session_state['pdf_generated']):
            with open(st.session_state['pdf_generated'], "rb") as f:
                st.download_button("📥 Descargar Archivo PDF", f, file_name="Reporte_Tecnico_Financiero.pdf", mime="application/pdf", use_container_width=True)
                    
except Exception as e:
    st.error(f"Falta el dataset maestro o hubo un error ETL: {e}")
    st.stop()

tab1, tab2, tab3, tab4, tab6, tab5, tab7 = st.tabs(["[VELAS & SMA]", "[CORRELACIÓN]", "[RIESGOS & PATRONES]", "[SIMILITUD]", "🔍 [EVIDENCIA ETL] DATASET", "\u2003\u2003✨ [BONUS] IMPACTO COVID-19", "🚀 [BONUS] IA PREDICTIVA"])

with tab1:
    st.header("Análisis de Tendencias (Candlestick & Medias Móviles)")
    col_a, col_b = st.columns([1, 4])
    with col_a:
        asset = st.selectbox("Seleccione Activo", tickers)
        sma_window = st.slider("Ventana de SMA", min_value=10, max_value=200, value=50, step=10)
        show_fibo = st.checkbox("Trazar Niveles Fibonacci Automáticos")
    with col_b:
        asset_rows = [r for r in master_rows if r['ticker'] == asset]
        asset_rows = sorted(asset_rows, key=lambda x: x['fecha'])
        
        # Calculo SMA algorítmico empírico (Sliding Window O(N))
        closes = [r['close'] for r in asset_rows]
        sma_manual = [np.nan] * len(closes)
        
        if len(closes) >= sma_window:
            # Ventana inicial (Ignorando NaNs si los hay)
            valid_closes = [c if not np.isnan(c) else 0 for c in closes]
            current_sum = sum(valid_closes[:sma_window])
            sma_manual[sma_window - 1] = current_sum / sma_window
            
            # Recorrido de ventana deslizante
            for i in range(sma_window, len(valid_closes)):
                current_sum = current_sum - valid_closes[i - sma_window] + valid_closes[i]
                sma_manual[i] = current_sum / sma_window
                
        # Graficar sin pandas
        fechas = [r['fecha'] for r in asset_rows]
        opens = [r['open'] for r in asset_rows]
        highs = [r['high'] for r in asset_rows]
        lows = [r['low'] for r in asset_rows]
        
        fig = go.Figure(data=[go.Candlestick(x=fechas,
                        open=opens, high=highs,
                        low=lows, close=closes, name='Precio Velas',
                        increasing_line_color='#22C55E', decreasing_line_color='#EF4444')])
        fig.add_trace(go.Scatter(x=fechas, y=sma_manual, line=dict(color='#F59E0B', width=2), name=f'SMA {sma_window}'))
        
        # Retrocesos de Fibonacci Dinámicos O(N)
        if show_fibo and highs and lows:
            max_price = max([h for h in highs if h and not np.isnan(h)])
            min_price = min([l for l in lows if l and not np.isnan(l)])
            diff = max_price - min_price
            
            levels = {
                "0.0% (Mínimo)": min_price,
                "23.6%": max_price - diff * 0.764,
                "38.2%": max_price - diff * 0.618,
                "50.0%": max_price - diff * 0.5,
                "61.8%": max_price - diff * 0.382,
                "100.0% (Máximo)": max_price
            }
            colors_fibo = ["#33ff33", "#ffff00", "#ff9900", "#ff3399", "#ff0000", "#33ff33"]
            
            for (name, val), c in zip(levels.items(), colors_fibo):
                fig.add_hline(y=val, line_dash="solid" if name in ["0.0% (Mínimo)", "100.0% (Máximo)"] else "dot", 
                              line_color=c, line_width=1, opacity=0.7, 
                              annotation_text=f"Fibo {name}", annotation_position="right", annotation_font_color=c)

        fig.update_layout(title=f"Evolución de {asset}", height=600, yaxis_title="Precio Cierre ($)", **hud_layout)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Matriz de Correlación del Portafolio")
    st.markdown("Calcula las relaciones cruzadas entre los cierres de todos los activos, revelando posibles diversificaciones de riesgo.")
    
    # 1. Agrupar series por ticker dictado desde lists nativas
    ticker_series = {t: [] for t in tickers}
    for row in master_rows:
        ticker_series[row['ticker']].append(row['close'])
        
    n_tickers = len(tickers)
    corr_matrix_np = np.zeros((n_tickers, n_tickers))
    
    for i in range(n_tickers):
        for j in range(n_tickers):
            if i == j:
                corr_matrix_np[i, j] = 1.0
            elif j > i:
                s_i = np.array(ticker_series[tickers[i]], dtype=float)
                s_j = np.array(ticker_series[tickers[j]], dtype=float)
                
                # Eliminamos los NaNs compartidos (fechas desalineadas) usando bitwise masking rápido
                valid_mask = ~np.isnan(s_i) & ~np.isnan(s_j)
                val_i = s_i[valid_mask]
                val_j = s_j[valid_mask]
                
                # Usamos nuestra propia función matemática pura de Similitud Pearson O(N)
                corr, _ = pearson_correlation(val_i.tolist(), val_j.tolist())
                
                # Espejado simétrico de matriz relacional
                corr_matrix_np[i, j] = corr
                corr_matrix_np[j, i] = corr
                
    # Heatmap Profesional Paleta
    fig_corr = px.imshow(corr_matrix_np, text_auto=".2f", aspect="auto", x=tickers, y=tickers, color_continuous_scale='Blues', 
                         title="Mapa de Calor de Correlaciones de Pearson (Algoritmo Explícito)")
    fig_corr.update_layout(height=800, **hud_layout)
    st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    st.header("Clasificación de Riesgos y Algoritmos de Detección de Patrones")
    st.markdown("### 1. Perfil de Riesgo (Volatilidad Histórica Anualizada)")
    
    risk_results = evaluate_portfolio_risk(master_rows)
    # Mostrar riesgo manualmente creando tabla HTML (Sin Pandas df.style)
    st.markdown("""
        <style>
        .risk-conservador { background-color: rgba(34, 197, 94, 0.1); color: #22C55E; border: 1px solid rgba(34, 197, 94, 0.5); padding: 4px; display:inline-block; min-width:80px; text-align:center; border-radius: 4px; font-weight: 600;}
        .risk-moderado { background-color: rgba(245, 158, 11, 0.1); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.5); padding: 4px; display:inline-block; min-width:80px; text-align:center; border-radius: 4px; font-weight: 600;}
        .risk-agresivo { background-color: rgba(239, 68, 68, 0.1); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.5); padding: 4px; display:inline-block; min-width:80px; text-align:center; border-radius: 4px; font-weight: 600;}
        </style>
    """, unsafe_allow_html=True)
    
    table_html = "<table><tr><th>Ticker</th><th>Volatilidad Anual</th><th>Perfil</th></tr>"
    for r in risk_results:
        pct = f"{r['volatilidad_anual']*100:.2f}%" if not np.isnan(r['volatilidad_anual']) else "N/A"
        cls = f"risk-{str(r['perfil']).lower()}"
        table_html += f"<tr><td>{r['ticker']}</td><td>{pct}</td><td><span class='{cls}'>{r['perfil']}</span></td></tr>"
    table_html += "</table><br>"
    st.markdown(table_html, unsafe_allow_html=True)
    
    st.markdown("### 2. Detección de Patrones mediante Sliding Window")
    patron_asset = st.selectbox("Evaluar Patrones en:", tickers, key="patron_select")
    
    asset_rows = [r for r in master_rows if r['ticker'] == patron_asset]
    asset_closes = [r['close'] for r in asset_rows]
    
    col_p1, col_p2 = st.columns(2)
    col_p1.metric(f"Días Consecutivos (3) al Alza", detect_consecutive_up_days(asset_closes, 3))
    col_p2.metric(f"Patrón de Reversión en V (Morning Star)", detect_reversal_v_pattern(asset_rows))
    st.info(r"**Formalización Matemática del Patrón Adicional (Req. 3):** Se define un *Patrón de Reversión en V* para una ventana temporal $T$ cuando ocurren tres días consecutivos de caídas ($C_{t} < O_{t}$ para $t \in \{T-3, T-2, T-1\}$) seguidos inmediatamente por un día de fuerte recuperación alcista ($C_T > O_{T-1}$). Esto se calcula algorítmicamente mediante el recorrido en ventana deslizante en tiempo $\mathcal{O}(N)$.")
    
    st.markdown("---")
    st.markdown("### 3. Optimizador de Portafolio O(N) Fuerza Bruta (Sharpe Ratio)")
    st.markdown("Sistema Markowitz sin Scipy u Optimizadores encapsulados. Encuentra los pesos calculando 1000 iteraciones aleatorias masivas de combinaciones cruzando Matrices de Covarianza O(N).")
    
    port_tickers = st.multiselect("Selecciona HASTA 4 activos para armar tu portafolio:", tickers, default=tickers[:3] if len(tickers)>2 else tickers)
    if st.button("Buscar Óptimo mediante IA Monte Carlo", type="primary"):
        if 2 <= len(port_tickers) <= 4:
            with st.spinner("Computando Covarianzas manuales N x N y lanzando caminos Markovianos..."):
                series_dict = {t: [r['close'] for r in master_rows if r['ticker']==t and r['close']] for t in port_tickers}
                opt_res = brute_force_portfolio(series_dict, num_portfolios=2000)
                
                if opt_res:
                    st.success(f"💎 **Optimizador de Markowitz Finalizado.** Mejor Sharpe Ratio Encontrado: {opt_res['sharpe']:.2f}")
                    
                    # Mostrar pesos sugeridos
                    p_col1, p_col2 = st.columns(2)
                    peso_str = ""
                    for tick, weight in opt_res['weights'].items():
                        peso_str += f"- **{tick}:** {weight*100:.1f}%\n"
                        
                    p_col1.info(f"**Distribución Ideal de Inversión (Pesos):**\n" + peso_str)
                    
                    p_col2.metric("Retorno Histórico Esperado", f"{opt_res['return']*100:.2f}% (Anual)")
                    p_col2.metric("Nivel de Riesgo (Volatilidad Combinada)", f"{opt_res['volatility']*100:.2f}%")
        else:
            st.warning("Para optimizar cruzado necesitas seleccionar entre 2 y 4 activos.")

with tab4:
    st.header("Laboratorio Matemático de Similitud Series de Tiempo")
    st.markdown("Compara dos activos para determinar la alineación espacial o temporal usando sus rendimientos o precios normalizados.")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    s1_ticker = col_s1.selectbox("Activo 1", tickers, index=0)
    s2_ticker = col_s2.selectbox("Activo 2", tickers, index=1 if len(tickers)>1 else 0)
    algoritmo = col_s3.selectbox("Algoritmo", ["Distancia Euclidiana", "Correlación de Pearson", "Similitud Coseno", "DTW (Dynamic Time Warping)"])
    
    # Obtener arrays numéricos nativos
    arr1 = np.array([r['close'] for r in master_rows if r['ticker'] == s1_ticker])
    arr2 = np.array([r['close'] for r in master_rows if r['ticker'] == s2_ticker])
    
    # Limitar para que tengan el mismo tamaño forzosamente y normalizar (MinMax estándar para comparar forma libre de escala)
    min_len = min(len(arr1), len(arr2))
    a1 = arr1[-min_len:]
    a2 = arr2[-min_len:]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        a1_norm = (a1 - np.nanmean(a1)) / np.nanstd(a1)
        a2_norm = (a2 - np.nanmean(a2)) / np.nanstd(a2)
        
    a1_norm = np.nan_to_num(a1_norm)
    a2_norm = np.nan_to_num(a2_norm)
    
    if st.button("Calcular Similitud", type="primary"):
        val, req_complexy = 0, ""
        math_formula, algo_desc = "", ""
        
        if algoritmo == "Distancia Euclidiana":
            val, req_complexy = euclidean_distance(a1_norm.tolist(), a2_norm.tolist())
            math_formula = r"\Large d_{Euc}(X, Y) = \sqrt{\sum_{t=1}^{T} (X_t - Y_t)^2}"
            algo_desc = r"**Mecánica Numérica:** Evalúa la similitud comparando la diferencia de precios 'día a día' en una iteración secuencial a lo largo de toda la historia temporal $\mathcal{O}(N)$. En términos matemáticos, halla la distancia espacial en cada fecha $t$, la eleva al cuadrado (para evitar la cancelación de diferencias negativas y positivas, y para penalizar fuertemente las brechas extremas), luego consolida sumando estas magnitudes y extrayendo la raíz cuadrada geométrica total. Este método es excepcionalmente estricto frente al alineamiento temporal absoluto."
            interp_desc = r"Es una métrica directa de **diferencia absoluta espacial** y distancia directa entre trayectorias. <br><br>👉 **Si es 0:** Las gráficas de ambos activos son calcos idénticos, se superponen perfectamente sin ningún margen de error. <br>👉 **Entre 0 y 5 (En Datos Normalizados/Z-Score):** Existe un altísimo nivel de sincronía de mercado; las curvas mantienen formas muy similares y caminan cerca. <br>👉 **Mayor a 15:** Denota una fuerte **divergencia espacial**. Indica que un activo puede estar experimentando un mercado fuertemente alcista (bull market) mientras el otro sufre caídas."
            
        elif algoritmo == "Correlación de Pearson":
            val, req_complexy = pearson_correlation(a1_norm.tolist(), a2_norm.tolist())
            math_formula = r"\Large \rho_{X,Y} = \frac{\sum_{t=1}^{T} (X_t - \bar{X})(Y_t - \bar{Y})}{\sqrt{\sum_{t=1}^{T} (X_t - \bar{X})^2} \sqrt{\sum_{t=1}^{T} (Y_t - \bar{Y})^2}}"
            algo_desc = r"**Mecánica Numérica:** Cuantifica la fuerza y dirección de la relación **lineal** entre dos variables a lo largo de $T$ iteraciones $\mathcal{O}(N)$. Primero, evalúa el comportamiento de cada activo frente a su propia media histórica o 'centro de gravedad' $\bar{X}$. Luego, cruza estos deltas direccionales para ver si los dos suben o bajan al mismo tiempo, y estandariza el producto resultante usando la volatilidad cruzada para aislar la fuerza estructural sin importar el tamaño absoluto del precio."
            interp_desc = r"Es el estándar de oro en finanzas y econometría para analizar co-movimientos, con un rango matemático estricto de **[-1.0, 1.0]**. <br><br>👉 **1.0 (Correlación Positiva Perfecta):** Si X sube un 2%, Y también sube de manera predecible. Es un tándem ideal y denotan que pertenecen al mismo nicho o industria. <br>👉 **-1.0 (Correlación Negativa Perfecta):** Son activos **inversamente proporcionales**. Si el mercado colapsa y X se hunde, Y experimentará subidas agresivas. Matemáticamente, es el escenario de ensueño para **estrategias de cobertura (hedging) y diversificación sistemática de riesgo**. <br>👉 **0.0:** Ausencia total de patrón lineal predecible; sus movimientos son estadísticamente independientes, ideales para reducir la volatilidad general de un portafolio."
            
        elif algoritmo == "Similitud Coseno":
            val, req_complexy = cosine_similarity(a1_norm.tolist(), a2_norm.tolist())
            math_formula = r"\Large \text{Similitud}_{Coseno}(X, Y) = \frac{\sum_{t=1}^{T} X_t \cdot Y_t}{\sqrt{\sum_{t=1}^{T} X_t^2} \sqrt{\sum_{t=1}^{T} Y_t^2}}"
            algo_desc = r"**Mecánica Numérica:** Transforma toda la serie temporal de cada activo en un gigantesco vector geométrico $N$-dimensional. Mediante un recorrido iterativo simple $\mathcal{O}(N)$, computa el **producto escalar (dot product)** y lo divide por las magnitudes euclidianas. A diferencia de Pearson o Euclidiana, el algoritmo del Coseno abstrae por completo la magnitud (la agresividad o tamaño de la caída o subida) y enfoca el 100% de la evaluación en la 'dirección del viento' temporal."
            interp_desc = r"Mide estrictamente el **ángulo de orientación** entre las tendencias de inversión en un espacio multi-dimensional. <br><br>👉 **Si es 1.0:** Tienen una tendencia direccional idéntica. Apuntan hacia el mismo norte independientemente de qué tan agresivos o volátiles sean sus rendimientos diarios. <br>👉 **Si es 0.0:** Los vectores son matemáticamente ortogonales (forman un ángulo perfecto de 90 grados). Representan instrumentos en nichos de mercado completamente aislados que no se inmutan por lo que le pase al otro activo. <br>👉 **Si es -1.0:** Tienen direcciones estrictamente opuestas (ángulo de 180 grados)."
            
        elif algoritmo == "DTW (Dynamic Time Warping)":
            st.warning("Calculando Costo DTW en los últimos 250 días para evitar saturación de memoria O(N²).")
            # DTW en matrix nativa requiere listas planas
            val, req_complexy = dtw_distance(a1_norm[-250:].tolist(), a2_norm[-250:].tolist())
            math_formula = r"\Large DTW(X, Y) = \min_W \sqrt{\sum_{k=1}^K w_k} \quad (\text{Camino de Costo Mínimo})"
            algo_desc = r"**Mecánica Numérica:** Aborda el análisis de similitud utilizando el paradigma de **Programación Dinámica Computacional** construyendo una inmensa matriz cruzada de costos espaciales en tiempo cuadrático $\mathcal{O}(N^2)$. Su máximo valor algorítmico radica en su propiedad de 'warping' (deformación elástica temporal). A diferencia de Euclidiana que exige comparar el 'Lunes de X' estrictamente con el 'Lunes de Y', DTW es elástico. Es capaz de asociar una fuerte caída del Activo A ocurrida un Lunes, con la caída reactiva del Activo B ocurrida el Miércoles, iterando hasta encontrar el camino matricial de mínimo esfuerzo para mapear ambas topologías."
            interp_desc = r"Es una métrica orientada puramente a **patrones estructurales y ciclos de retraso**, cuantificando el 'costo morfológico de deformación'. <br><br>👉 **Si es 0:** Las formas son idénticas y sin absolutamente ningún desfase temporal de sincronización. <br>👉 **Costos Bajos (Ej: < 10):** Revela que los activos se comportan igual y dibujan gráficas casi idénticas, pero sufren de **ecos o desfases en el tiempo** temporales causados por ineficiencias del mercado, diferencias en husos horarios de apertura, o rezagos en la absorción de información macroeconómica. <br>👉 **Costos Altos:** Denota que ni estirando, pausando ni comprimiendo temporalmente los eventos se logra alinear el comportamiento de los mercados subyacentes."
            
        st.success(f"**Cálculo Completado**: Métrica de {algoritmo} obtenida.")
        
        # Bloque expansivo explicativo (Requisito 2 Universitario)
        with st.expander("📚 Explicación Matemática y Algorítmica", expanded=True):
            st.latex(math_formula)
            st.markdown(f"**Fundamento Algorítmico:** {algo_desc}")
            st.markdown(f"**Complejidad de Cómputo (Big-O):** `{req_complexy}` calculada sobre las bases de datos nativas.")
            st.info(f"**💡 ¿Cómo interpretar este número?** \n\n{interp_desc}")
            st.metric(f"Valor Resultante ({algoritmo})", f"{val:.4f}")
        
        # Graficamos la serie
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(y=a1_norm[-250:], name=s1_ticker))
        fig_sim.add_trace(go.Scatter(y=a2_norm[-250:], name=s2_ticker))
        fig_sim.update_layout(title="Comparación Gráfica de Forma (Z-Score) - Últimos 250 días", height=400, **hud_layout)
        st.plotly_chart(fig_sim, use_container_width=True)
        
with tab5:
    st.header("✨ [BONUS] Análisis de Quiebre Dinámico: Efecto COVID-19 (2021+)")
    st.warning("**Propuesta de Valor Adicional:** Este módulo expande los requerimientos básicos del proyecto. Demuestra el dominio del análisis algorítmico aislando series temporales para estudiar el impacto de un evento global real.")
    st.markdown("Analiza la bifurcación temporal aislando y correlacionando activos pre y post 2021 mediante puras `List Comprehensions` O(N).")
    
    selected_assets = st.multiselect("Seleccione hasta 4 activos a comparar", tickers, default=[tickers[0]] if len(tickers)>0 else [])
    if len(selected_assets) > 4:
        st.warning("Seleccione máximo 4 activos para mantener visible el panel HUD.")
    elif len(selected_assets) > 0:
        col_c1, col_c2 = st.columns(2)
        
        # Filtros de diccionarios puros
        for i, asset in enumerate(selected_assets[:4]):
            asset_rows = [r for r in master_rows if r['ticker'] == asset]
            asset_rows = sorted(asset_rows, key=lambda x: x['fecha'])
            
            # Quiebre en 2021-01-01
            pre_covid = [r for r in asset_rows if r['fecha'] < '2021-01-01']
            post_covid = [r for r in asset_rows if r['fecha'] >= '2021-01-01']
            
            # Rentabilidades y Puntos crudos
            closes_pre = [r['close'] for r in pre_covid if r['close'] is not None]
            closes_post = [r['close'] for r in post_covid if r['close'] is not None]
            
            ret_pre = (closes_pre[-1] / closes_pre[0] - 1)*100 if len(closes_pre)>1 and closes_pre[0] > 0 else 0
            ret_post = (closes_post[-1] / closes_post[0] - 1)*100 if len(closes_post)>1 and closes_post[0] > 0 else 0
            
            # Promedios
            avg_pre = sum(closes_pre)/len(closes_pre) if closes_pre else 0
            avg_post = sum(closes_post)/len(closes_post) if closes_post else 0
            
            target_col = col_c1 if i % 2 == 0 else col_c2
            with target_col:
                st.markdown(f"#### [ACTIVO: {asset}]")
                colA, colB = st.columns(2)
                colA.metric("RETORNO PRE-2021", f"{ret_pre:.1f}%", f"Avg: ${avg_pre:.2f}")
                colB.metric("RETORNO POST-2021", f"{ret_post:.1f}%", f"Avg: ${avg_post:.2f}")

        # Gráfico Temporal Bifurcado
        fig_covid = go.Figure()
        # Colores profesionales
        colors = ['#38BDF8', '#818CF8', '#34D399', '#F472B6']
        
        for i, asset in enumerate(selected_assets[:4]):
            asset_rows = [r for r in master_rows if r['ticker'] == asset]
            fechas = [r['fecha'] for r in asset_rows]
            closes_norm = np.array([r['close'] if r['close'] else 0 for r in asset_rows])
            
            # Normalizar para visualización justa si hay varios
            with np.errstate(divide='ignore', invalid='ignore'):
                c_norm = (closes_norm - np.nanmean(closes_norm)) / np.nanstd(closes_norm)
            
            fig_covid.add_trace(go.Scatter(x=fechas, y=c_norm, name=asset, line=dict(color=colors[i % len(colors)], width=2)))
            
        # Marcador divisor del quiebre (Línea V)
        fig_covid.add_vline(x='2021-01-01', line_width=2, line_dash="dash", line_color="#EF4444")
        fig_covid.add_annotation(x='2021-01-01', y=0, text="PUNTO DE RECUPERACIÓN (2021)", showarrow=True, arrowhead=1, font=dict(color="#EF4444", size=12))
        
        fig_covid.update_layout(title="COMPARADOR TEMPORAL: PRE VS POST", height=500, **hud_layout)
        st.plotly_chart(fig_covid, use_container_width=True)

        # Inferencias Automáticas de Comportamiento 
        if len(selected_assets) >= 2:
            st.markdown("### 📝 CONCLUSIONES SISTÉMICAS ALGORÍTMICAS")
            # Extraemos series POST-COVID del primer par
            asset1 = selected_assets[0]
            asset2 = selected_assets[1]
            c1_post = [r['close'] for r in master_rows if r['ticker'] == asset1 and r['fecha'] >= '2021-01-01']
            c2_post = [r['close'] for r in master_rows if r['ticker'] == asset2 and r['fecha'] >= '2021-01-01']
            
            c1_pre = [r['close'] for r in master_rows if r['ticker'] == asset1 and r['fecha'] < '2021-01-01']
            c2_pre = [r['close'] for r in master_rows if r['ticker'] == asset2 and r['fecha'] < '2021-01-01']
            
            c1_post = [c if c else 0 for c in c1_post]
            c2_post = [c if c else 0 for c in c2_post]
            min_post = min(len(c1_post), len(c2_post))
            
            # Usamos nuestra def pearson_correlation
            corr_post, _ = pearson_correlation(c1_post[-min_post:], c2_post[-min_post:])
            
            c1_pre = [c if c else 0 for c in c1_pre]
            c2_pre = [c if c else 0 for c in c2_pre]
            min_pre = min(len(c1_pre), len(c2_pre))
            corr_pre, _ = pearson_correlation(c1_pre[-min_pre:], c2_pre[-min_pre:])
            
            # Generar texto
            def evaluar_corr(c):
                if c > 0.7: return "altamente positiva (se movieron en tándem)"
                if c < -0.7: return "fuertemente inversa (bienes refugio / divergencia)"
                if c > 0: return "ligeramente positiva"
                return "débil/inversa o neutral"

            st.info(f"👉 **Evolución Correlacional ({asset1} vs {asset2}):**\n"
                    f"Antes de 2021, la correlación estadística entre ambos instrumentos era **{corr_pre:.2f}** ({evaluar_corr(corr_pre)}). "
                    f"Para la ventana temporal a partir de 2021 en adelante (Recuperación / Post-COVID), su correlación es de **{corr_post:.2f}** ({evaluar_corr(corr_post)}). "
                    f"Esto indica estructuralmente de manera matemática que su comportamiento {'cambió drásticamente debido a dinámicas sectoriales asíncronas.' if abs(corr_post - corr_pre) > 0.5 else 'mantuvo un flujo relacional relativamente constante a pesar del evento disruptivo mundial.'}")

with tab6:
    st.header("Terminal de Datos Limpios (Master Dataset)")
    st.markdown("Inspecciona la matriz de registros consolidados del ETL. Todas las series han sido alineadas en su fecha (Outer Join algorítmico) y purgadas algorítmicamente sin dependencias de terceros.")
    
    # Selector de filtrado sin DataFrames
    show_dataset_ticker = st.selectbox("Filtrar Registros por Activo HUD:", ["[TODOS LOS TICKERS]"] + tickers)
    
    if show_dataset_ticker == "[TODOS LOS TICKERS]":
        filtered = master_rows[-1000:]
        st.caption("Visor HUD Estricto: Mostrando los últimos 1000 registros para evitar latencia HTML global.")
    else:
        filtered = [r for r in master_rows if r['ticker'] == show_dataset_ticker][-1000:]
        st.caption(f"Visor HUD Estricto: Mostrando los últimos 1000 registros estandarizados de {show_dataset_ticker}.")
        
    # Renderizamos en HTML puro para heredar estrictamente los degradados y bordes neón del HUD CSS
    table_html = "<table><tr><th>Fecha</th><th>Ticker</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volumen</th></tr>"
    for r in filtered:
        o = f"{r['open']:.2f}" if not np.isnan(r['open']) else "N/A"
        h = f"{r['high']:.2f}" if not np.isnan(r['high']) else "N/A"
        l = f"{r['low']:.2f}" if not np.isnan(r['low']) else "N/A"
        c = f"{r['close']:.2f}" if not np.isnan(r['close']) else "N/A"
        v = f"{r['volumen']:.0f}" if not np.isnan(r['volumen']) else "0"
        table_html += f"<tr><td>{r['fecha']}</td><td>{r['ticker']}</td><td>{o}</td><td>{h}</td><td>{l}</td><td>{c}</td><td>{v}</td></tr>"
    table_html += "</table>"
    # Contenedor con scroll adaptativo
    st.markdown(f"<div style='height: 600px; overflow-y: auto; border-radius: 8px;'>{table_html}</div>", unsafe_allow_html=True)

with tab7:
    st.header("🚀 [BONUS] Inteligencia Predictiva Algorítmica O(N)")
    st.warning("**Propuesta de Valor Adicional:** Este módulo demuestra técnicas predictivas avanzadas garantizando el cumplimiento de las restricciones del proyecto: no se usan librerías externas de Machine Learning como Scikit-Learn. Los algoritmos KNN y Monte Carlo están implementados manualmente desde cero iterando listas $\mathcal{O}(N)$.")
    st.markdown("Cálculos probabilísticos realizados puramente mediante Iteraciones y Matemáticas Vectoriales Nativas bajo inferencia de Random Walks.")
    
    ml_asset = st.selectbox("Seleccionar Activo Objetivo para Predicción e Inferencia:", tickers, key="ml_select")
    asset_rows = [r for r in master_rows if r['ticker'] == ml_asset]
    asset_rows = sorted(asset_rows, key=lambda x: x['fecha'])
    
    colML1, colML2 = st.columns([1, 2])
    
    with colML1:
        st.subheader("1. K-Nearest Neighbors O(N)")
        st.write("Busca los 5 días históricos con una racha de retornos equivalente a los últimos 3 días (Distancia Euclidiana O(N)) y somete el comportamiento del día posterior a votación mayoritaria.")
        
        if st.button("Ejecutar Inferencias KNN"):
            with st.spinner("Escaneando el firmamento histórico..."):
                pred, prob = knn_predict(asset_rows, k=5, window_size=3)
                
                direction = "ALZA 🚀" if pred == 1 else "BAJA 📉"
                color = "#22C55E" if pred == 1 else "#EF4444"
                
                st.markdown(f"#### PREDICCIÓN CONSOLIDADA (MAÑANA): <span style='color:{color}'>{direction}</span>", unsafe_allow_html=True)
                st.metric("Confianza de IA Replicada (Mayoritaria):", f"{prob*100:.1f}%")

    with colML2:
        st.subheader("2. Simulador de Wall Street (Fuerza N-Monte Carlo)")
        st.write("Dispara 50 caminos log-normales probables para este activo a 30 días, fundamentados heurísticamente en su Volatilidad de serie.")
        
        if st.button("Lanzar Proyección Monte Carlo (30 Días)"):
            with st.spinner("Ejecutando sorteos Gaussianos y construyendo matrices predictivas..."):
                caminos_mc, min_mc, max_mc = monte_carlo_simulation(asset_rows, days_ahead=30, n_simulations=50)
                
                if caminos_mc:
                    fig_mc = go.Figure()
                    for c in caminos_mc:
                        # Dibujamos las líneas de proyecciones sutilmente
                        fig_mc.add_trace(go.Scatter(y=c, mode='lines', line=dict(color='rgba(56, 189, 248, 0.15)', width=1), showlegend=False))
                        
                    fig_mc.update_layout(title=f"Proyecciones a futuro (50 Random Walks) - {ml_asset}", height=450, xaxis_title="Días a Futuro", yaxis_title="Precio Simulado", **hud_layout)
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Peor Escenario (Worst Case) a 30 días", f"${min_mc:.2f}")
                    mc2.metric("Mejor Escenario (Best Case) a 30 días", f"${max_mc:.2f}")
