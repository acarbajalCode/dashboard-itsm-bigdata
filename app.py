import streamlit as st
import pymongo
import pandas as pd
import plotly.express as px

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Dashboard Gerencial ITSM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNCIÓN DE CARGA Y PREPROCESAMIENTO ---
@st.cache_data(ttl=300)
def cargar_datos_mongo():
    cliente = pymongo.MongoClient(st.secrets["MONGO_URI"])
    db = cliente["itsm_analytics"]
    
    df_top = pd.DataFrame(list(db["top_incidentes"].find())).drop(columns=["_id"], errors='ignore')
    df_brechas = pd.DataFrame(list(db["brechas_servicio"].find())).drop(columns=["_id"], errors='ignore')
    df_tendencias = pd.DataFrame(list(db["tendencias_temporales"].find())).drop(columns=["_id"], errors='ignore')
    
    for df in [df_top, df_brechas, df_tendencias]:
        if not df.empty and 'Anio_Mes' in df.columns:
            df['Anio'] = df['Anio_Mes'].str.split('-').str[0]
            df['Mes'] = df['Anio_Mes'].str.split('-').str[1]
            
    return df_top, df_brechas, df_tendencias

try:
    df_top_raw, df_brechas_raw, df_tendencias_raw = cargar_datos_mongo()

    # --- PANEL LATERAL DE FILTROS (SLICERS GLOBALES) ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1828/1828859.png", width=50)
    st.sidebar.title("🎛️ Filtros Globales")
    st.sidebar.markdown("**Fase de pruebas tipo QA**")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### 📅 Filtro de Tiempo")
    anios_disponibles = sorted(df_tendencias_raw['Anio'].unique().tolist()) if not df_tendencias_raw.empty and 'Anio' in df_tendencias_raw.columns else []
    anio_sel = st.sidebar.selectbox("1. Seleccione el Año:", ["Todos"] + anios_disponibles)
    
    if anio_sel != "Todos":
        df_meses_filtrados = df_tendencias_raw[df_tendencias_raw['Anio'] == anio_sel]
        meses_disponibles = sorted(df_meses_filtrados['Mes'].unique().tolist())
    else:
        meses_disponibles = sorted(df_tendencias_raw['Mes'].unique().tolist()) if not df_tendencias_raw.empty and 'Mes' in df_tendencias_raw.columns else []

    mes_sel = st.sidebar.selectbox("2. Seleccione el Mes:", ["Todos"] + meses_disponibles)
    st.sidebar.markdown("---")

    st.sidebar.markdown("### 💻 Filtro de Infraestructura")
    hardware_disponible = sorted(df_brechas_raw["tipo_hardware"].unique().tolist()) if not df_brechas_raw.empty and "tipo_hardware" in df_brechas_raw.columns else []
    hardware_sel = st.sidebar.multiselect("Tipo de Hardware:", options=hardware_disponible, default=hardware_disponible)

    garantia_disponible = sorted(df_brechas_raw["estado_garantia"].unique().tolist()) if not df_brechas_raw.empty and "estado_garantia" in df_brechas_raw.columns else []
    garantia_sel = st.sidebar.multiselect("Estado de Garantía:", options=garantia_disponible, default=garantia_disponible)

    # --- MOTOR DE FILTRADO CRUZADO ---
    df_tendencias = df_tendencias_raw.copy()
    df_brechas = df_brechas_raw.copy()
    df_top = df_top_raw.copy()

    if anio_sel != "Todos":
        if 'Anio' in df_tendencias.columns: df_tendencias = df_tendencias[df_tendencias['Anio'] == anio_sel]
        if 'Anio' in df_brechas.columns: df_brechas = df_brechas[df_brechas['Anio'] == anio_sel]
        if 'Anio' in df_top.columns: df_top = df_top[df_top['Anio'] == anio_sel]

    if mes_sel != "Todos":
        if 'Mes' in df_tendencias.columns: df_tendencias = df_tendencias[df_tendencias['Mes'] == mes_sel]
        if 'Mes' in df_brechas.columns: df_brechas = df_brechas[df_brechas['Mes'] == mes_sel]
        if 'Mes' in df_top.columns: df_top = df_top[df_top['Mes'] == mes_sel]

    if hardware_sel:
        if 'tipo_hardware' in df_tendencias.columns: df_tendencias = df_tendencias[df_tendencias['tipo_hardware'].isin(hardware_sel)]
        if 'tipo_hardware' in df_brechas.columns: df_brechas = df_brechas[df_brechas['tipo_hardware'].isin(hardware_sel)]
        if 'tipo_hardware' in df_top.columns: df_top = df_top[df_top['tipo_hardware'].isin(hardware_sel)]

    if garantia_sel:
        if 'estado_garantia' in df_tendencias.columns: df_tendencias = df_tendencias[df_tendencias['estado_garantia'].isin(garantia_sel)]
        if 'estado_garantia' in df_brechas.columns: df_brechas = df_brechas[df_brechas['estado_garantia'].isin(garantia_sel)]
        if 'estado_garantia' in df_top.columns: df_top = df_top[df_top['estado_garantia'].isin(garantia_sel)]

    st.title("📊 Dashboard Ejecutivo ITSM & Analítica Predictiva")
    st.caption("Visión integrada de incidentes, infraestructura y satisfacción afectada por filtros globales")
    st.markdown("---")

    # --- PESTAÑAS Y NAVEGACIÓN ---
    pestana = st.radio(
        "Seleccione la vista gerencial:",
        ["📈 Visión General y Tendencias", "🛡️ Análisis de Brechas y Hardware", "🔎 Detalle y Drill-Down de Equipos"],
        horizontal=True
    )
    st.markdown("---")

    # --- VISTA 1: TENDENCIAS Y TOP INCIDENTES ---
    if pestana == "📈 Visión General y Tendencias":
        # KPIs exclusivos para la Vista 1
        tot_inc_tendencia = int(df_tendencias["Volumen_Mensual"].sum()) if not df_tendencias.empty else (int(df_top["Total_Incidentes"].sum()) if not df_top.empty else 0)
        
        k1, k2 = st.columns(2)
        k1.metric("📌 Total Incidentes (Histórico Filtrado)", f"{tot_inc_tendencia:,}")
        k2.metric("📋 Categorías Activas de Incidentes", f"{len(df_top) if not df_top.empty else 0}")
        st.markdown("<br>", unsafe_allow_html=True)

        col_izq, col_der = st.columns([6, 4])
        with col_izq:
            st.subheader(f"📈 Evolución Temporal ({anio_sel} - {mes_sel})")
            if not df_tendencias.empty and "Volumen_Mensual" in df_tendencias.columns:
                df_curva = df_tendencias.groupby("Anio_Mes")["Volumen_Mensual"].sum().reset_index()
                fig_linea = px.line(df_curva, x="Anio_Mes", y="Volumen_Mensual", markers=True, text="Volumen_Mensual")
                fig_linea.update_traces(textposition="top center", line_color="#0066CC")
                fig_linea.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=380)
                st.plotly_chart(fig_linea, use_container_width=True)
            else:
                st.info("No hay datos de tendencias para los filtros seleccionados.")
        with col_der:
            st.subheader("🔥 Top Incidentes (Filtrado)")
            if not df_top.empty and "Total_Incidentes" in df_top.columns:
                df_top_sub = df_top.head(8).sort_values(by="Total_Incidentes", ascending=True)
                fig_barras = px.bar(
                    df_top_sub, x="Total_Incidentes", y="Titulo_Limpio", orientation='h',
                    text="Total_Incidentes", color="Total_Incidentes", color_continuous_scale="Blues"
                )
                fig_barras.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=380, showlegend=False)
                st.plotly_chart(fig_barras, use_container_width=True)
            else:
                st.info("No hay datos de incidentes para los filtros seleccionados.")

    # --- VISTA 2: BRECHAS Y GARANTÍAS ---
    elif pestana == "🛡️ Análisis de Brechas y Hardware":
        # KPIs exclusivos para la Vista 2
        tot_inc_hw = int(df_brechas["Volumen_Incidentes"].sum()) if not df_brechas.empty else 0
        min_per = int(df_brechas["Minutos_Perdidos_Soporte"].sum()) if not df_brechas.empty else 0
        csat_prom = df_brechas["CSAT_Promedio"].mean() if not df_brechas.empty else 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📌 Tickets (Infraestructura)", f"{tot_inc_hw:,}")
        k2.metric("⭐ CSAT Promedio Global", f"{csat_prom:.2f} / 5.0")
        k3.metric("⏳ Soporte Perdido Total", f"{min_per:,} min")
        k4.metric("🖥️ Tipos Hardware Analizados", f"{len(df_brechas['tipo_hardware'].unique()) if not df_brechas.empty else 0}")
        st.markdown("<br>", unsafe_allow_html=True)

        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            st.subheader("⚖️ Volumen por Hardware (Filtrado)")
            if not df_brechas.empty:
                fig_hw = px.bar(
                    df_brechas, x="tipo_hardware", y="Volumen_Incidentes",
                    color="estado_garantia", barmode="group", text_auto=True,
                    color_discrete_map={"Vigente": "#28a745", "Vencida": "#dc3545"}
                )
                fig_hw.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=400)
                st.plotly_chart(fig_hw, use_container_width=True)
            else:
                st.warning("No hay datos para esta combinación de filtros.")
        with col_g2:
            st.subheader("⭐ CSAT vs Tiempo Perdido")
            if not df_brechas.empty:
                fig_bubble = px.scatter(
                    df_brechas, x="tipo_hardware", y="CSAT_Promedio",
                    size="Minutos_Perdidos_Soporte", color="estado_garantia",
                    hover_name="tipo_hardware",
                    color_discrete_map={"Vigente": "#28a745", "Vencida": "#dc3545"}
                )
                fig_bubble.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=400)
                st.plotly_chart(fig_bubble, use_container_width=True)

    # --- VISTA 3: DRILL-DOWN ---
    elif pestana == "🔎 Detalle y Drill-Down de Equipos":
        st.subheader("🔍 Explorador de Detalle")
        hardware_filtrado_disponible = sorted(df_brechas["tipo_hardware"].unique().tolist()) if not df_brechas.empty else []
        
        if not hardware_filtrado_disponible:
            st.warning("Los filtros actuales de la izquierda no dejaron ningún hardware disponible.")
        else:
            # Combo box interno
            equipo_seleccionado = st.selectbox("👉 Elija un Equipo Específico para ver su Ficha Técnica:", options=hardware_filtrado_disponible)
            
            if equipo_seleccionado:
                df_detalle_eq = df_brechas[df_brechas["tipo_hardware"] == equipo_seleccionado]
                
                # KPIs exclusivos, interactivos y ligados AL COMBO BOX
                tot_inc_eq = df_detalle_eq["Volumen_Incidentes"].sum()
                min_perd_eq = df_detalle_eq["Minutos_Perdidos_Soporte"].sum()
                csat_eq = df_detalle_eq["CSAT_Promedio"].mean()

                st.markdown(f"### 📋 Rendimiento Aislado: **{equipo_seleccionado}**")
                col_d1, col_d2, col_d3 = st.columns(3)
                col_d1.metric(f"📌 Incidentes ({equipo_seleccionado})", f"{tot_inc_eq:,}")
                col_d2.metric("⭐ CSAT Promedio", f"{csat_eq:.2f} / 5.0")
                col_d3.metric("⏳ Minutos Perdidos", f"{min_perd_eq:,} min")
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(df_detalle_eq.style.highlight_max(axis=0, color="#ffcccc"), use_container_width=True)

except Exception as e:
    st.error(f"❌ Error al conectar o cargar datos: {e}")
