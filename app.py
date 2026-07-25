import streamlit as st
import pymongo
import pandas as pd
import plotly.express as px

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Dashboard Gerencial ITSM - Power BI Style",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNCIÓN DE CARGA DE DATOS ---
@st.cache_data(ttl=300)
def cargar_datos_mongo():
    cliente = pymongo.MongoClient(st.secrets["MONGO_URI"])
    db = cliente["itsm_analytics"]
    
    df_top = pd.DataFrame(list(db["top_incidentes"].find())).drop(columns=["_id"], errors='ignore')
    df_brechas = pd.DataFrame(list(db["brechas_servicio"].find())).drop(columns=["_id"], errors='ignore')
    df_tendencias = pd.DataFrame(list(db["tendencias_temporales"].find())).drop(columns=["_id"], errors='ignore')
    
    # Preprocesamiento: Separar Anio_Mes en dos columnas para los Combos (Ej: "2023-07" -> "2023" y "07")
    if not df_tendencias.empty and "Anio_Mes" in df_tendencias.columns:
        df_tendencias['Anio'] = df_tendencias['Anio_Mes'].str.split('-').str[0]
        df_tendencias['Mes'] = df_tendencias['Anio_Mes'].str.split('-').str[1]
        
    return df_top, df_brechas, df_tendencias

try:
    df_top_raw, df_brechas_raw, df_tendencias_raw = cargar_datos_mongo()

    # --- PANEL LATERAL DE FILTROS (CASCADA TIPO POWER BI) ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1828/1828859.png", width=50)
    st.sidebar.title("🎛️ Filtros Interactivos")
    st.sidebar.markdown("---")

    # 1. FILTROS DE TIEMPO (COMBO BOXES EN CASCADA)
    st.sidebar.markdown("### 📅 Filtro de Tiempo")
    
    anios_disponibles = sorted(df_tendencias_raw['Anio'].unique().tolist()) if not df_tendencias_raw.empty else []
    
    # Combo 1: Seleccionar Año
    anio_sel = st.sidebar.selectbox("1. Seleccione el Año:", ["Todos"] + anios_disponibles)
    
    # Lógica de Cascada: Filtrar los meses dependiendo del Año seleccionado
    if anio_sel != "Todos":
        df_meses_filtrados = df_tendencias_raw[df_tendencias_raw['Anio'] == anio_sel]
        meses_disponibles = sorted(df_meses_filtrados['Mes'].unique().tolist())
    else:
        meses_disponibles = sorted(df_tendencias_raw['Mes'].unique().tolist()) if not df_tendencias_raw.empty else []

    # Combo 2: Seleccionar Mes (Se actualiza dinámicamente)
    mes_sel = st.sidebar.selectbox("2. Seleccione el Mes:", ["Todos"] + meses_disponibles)

    st.sidebar.markdown("---")

    # 2. FILTROS DE HARDWARE Y GARANTÍA (MULTIPLE SELECT PARA COMPARAR)
    st.sidebar.markdown("### 💻 Filtro de Infraestructura")
    hardware_disponible = sorted(df_brechas_raw["tipo_hardware"].unique().tolist()) if not df_brechas_raw.empty else []
    hardware_sel = st.sidebar.multiselect(
        "Tipo de Hardware:",
        options=hardware_disponible,
        default=hardware_disponible # Por defecto marca todos
    )

    garantia_disponible = sorted(df_brechas_raw["estado_garantia"].unique().tolist()) if not df_brechas_raw.empty else []
    garantia_sel = st.sidebar.multiselect(
        "Estado de Garantía:",
        options=garantia_disponible,
        default=garantia_disponible
    )

    # --- APLICACIÓN DE FILTROS A LOS DATAFRAMES ---
    
    # Aplicar filtros de tiempo a la tabla de tendencias
    df_tendencias = df_tendencias_raw.copy()
    if anio_sel != "Todos":
        df_tendencias = df_tendencias[df_tendencias['Anio'] == anio_sel]
    if mes_sel != "Todos":
        df_tendencias = df_tendencias[df_tendencias['Mes'] == mes_sel]

    # Aplicar filtros de hardware a la tabla de brechas
    df_brechas = df_brechas_raw[
        (df_brechas_raw["tipo_hardware"].isin(hardware_sel)) & 
        (df_brechas_raw["estado_garantia"].isin(garantia_sel))
    ] if not df_brechas_raw.empty else pd.DataFrame()
    
    df_top = df_top_raw if not df_top_raw.empty else pd.DataFrame()

    # --- ENCABEZADO Y KPIS DINÁMICOS ---
    st.title("📊 Dashboard Ejecutivo ITSM & Analítica Predictiva")
    st.caption("Visión integrada de incidentes, infraestructura y satisfacción en tiempo real")

    # Métricas dinámicas calculadas según los filtros
    tot_inc = int(df_tendencias["Volumen_Mensual"].sum()) if not df_tendencias.empty else 0
    if tot_inc == 0 and not df_top.empty: # Si el filtro de tiempo está vacío, mostramos el global
        tot_inc = int(df_top["Total_Incidentes"].sum())
        
    min_per = int(df_brechas["Minutos_Perdidos_Soporte"].sum()) if not df_brechas.empty else 0
    csat_prom = df_brechas["CSAT_Promedio"].mean() if not df_brechas.empty else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📌 Tickets en Periodo Seleccionado", f"{tot_inc:,}")
    k2.metric("⭐ CSAT Promedio", f"{csat_prom:.2f} / 5.0")
    k3.metric("⏳ Soporte Perdido (Min)", f"{min_per:,} min")
    k4.metric("🖥️ Tipos Hardware Evaluados", f"{len(hardware_sel)} / {len(hardware_disponible)}")

    st.markdown("---")

    # --- PESTAÑAS PRINCIPALES ---
    pestana = st.radio(
        "Seleccione la vista gerencial:",
        ["📈 Visión General y Tendencias", "🛡️ Análisis de Brechas y Hardware", "🔎 Detalle y Drill-Down de Equipos"],
        horizontal=True
    )

    st.markdown("---")

    # --- VISTA 1: TENDENCIAS Y TOP INCIDENTES ---
    if pestana == "📈 Visión General y Tendencias":
        col_izq, col_der = st.columns([6, 4])

        with col_izq:
            texto_titulo = f"📈 Evolución Temporal ({anio_sel} - {mes_sel})" if anio_sel != "Todos" else "📈 Evolución Temporal (Histórico Completo)"
            st.subheader(texto_titulo)
            
            if not df_tendencias.empty:
                df_curva = df_tendencias.groupby("Anio_Mes")["Volumen_Mensual"].sum().reset_index()
                fig_linea = px.line(
                    df_curva, x="Anio_Mes", y="Volumen_Mensual",
                    markers=True, text="Volumen_Mensual",
                    title="Tendencia de Tickets",
                    labels={"Anio_Mes": "Periodo", "Volumen_Mensual": "Tickets"}
                )
                fig_linea.update_traces(textposition="top center", line_color="#0066CC")
                fig_linea.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=380)
                st.plotly_chart(fig_linea, use_container_width=True)
            else:
                st.warning("No hay tickets reportados para el periodo seleccionado.")

        with col_der:
            st.subheader("🔥 Top Cuellos de Botella (Global)")
            if not df_top.empty:
                df_top_sub = df_top.head(8).sort_values(by="Total_Incidentes", ascending=True)
                fig_barras = px.bar(
                    df_top_sub, x="Total_Incidentes", y="Titulo_Limpio", orientation='h',
                    text="Total_Incidentes", color="Total_Incidentes",
                    color_continuous_scale="Blues",
                    labels={"Titulo_Limpio": "Incidente", "Total_Incidentes": "Cantidad"}
                )
                fig_barras.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=380, showlegend=False)
                st.plotly_chart(fig_barras, use_container_width=True)

    # --- VISTA 2: BRECHAS Y GARANTÍAS ---
    elif pestana == "🛡️ Análisis de Brechas y Hardware":
        col_g1, col_g2 = st.columns([1, 1])

        with col_g1:
            st.subheader("⚖️ Volumen de Incidentes por Tipo de Hardware")
            if not df_brechas.empty:
                fig_hw = px.bar(
                    df_brechas, x="tipo_hardware", y="Volumen_Incidentes",
                    color="estado_garantia", barmode="group",
                    text_auto=True,
                    color_discrete_map={"Vigente": "#28a745", "Vencida": "#dc3545"},
                    title="Comparativo: Garantía Vigente vs Vencida"
                )
                fig_hw.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=400)
                st.plotly_chart(fig_hw, use_container_width=True)
            else:
                st.warning("Seleccione al menos un tipo de hardware en el menú izquierdo.")

        with col_g2:
            st.subheader("⭐ Satisfacción (CSAT) vs Tiempo Perdido")
            if not df_brechas.empty:
                fig_bubble = px.scatter(
                    df_brechas, x="tipo_hardware", y="CSAT_Promedio",
                    size="Minutos_Perdidos_Soporte", color="estado_garantia",
                    hover_name="tipo_hardware",
                    title="Tamaño de burbuja = Minutos Perdidos en Soporte",
                    color_discrete_map={"Vigente": "#28a745", "Vencida": "#dc3545"}
                )
                fig_bubble.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=400)
                st.plotly_chart(fig_bubble, use_container_width=True)

    # --- VISTA 3: DRILL-DOWN / DETALLE POR EQUIPO (SELECCIÓN DINÁMICA) ---
    elif pestana == "🔎 Detalle y Drill-Down de Equipos":
        st.subheader("🔍 Explorador de Detalle (Ficha Técnica)")
        st.markdown("Seleccione un hardware específico para aislar su rendimiento:")

        # Combo selector de drill-down (Combo Box simple)
        equipo_seleccionado = st.selectbox(
            "👉 Elija un Equipo para ver su Ficha de Impacto:",
            options=hardware_disponible
        )

        if equipo_seleccionado:
            df_detalle_eq = df_brechas_raw[df_brechas_raw["tipo_hardware"] == equipo_seleccionado]

            st.markdown(f"### 📋 Ficha de Rendimiento: **{equipo_seleccionado}**")

            col_d1, col_d2, col_d3 = st.columns(3)
            tot_inc_eq = df_detalle_eq["Volumen_Incidentes"].sum()
            min_perd_eq = df_detalle_eq["Minutos_Perdidos_Soporte"].sum()
            csat_eq = df_detalle_eq["CSAT_Promedio"].mean()

            col_d1.metric(f"Incidentes ({equipo_seleccionado})", f"{tot_inc_eq:,}")
            col_d2.metric("Minutos de Soporte Consumidos", f"{min_perd_eq:,} min")
            col_d3.metric("Calificación CSAT", f"{csat_eq:.2f} / 5.0")

            st.markdown("#### Tabla Desglosada por Estado de Garantía:")
            st.dataframe(
                df_detalle_eq.style.highlight_max(axis=0, color="#ffcccc"),
                use_container_width=True
            )

except Exception as e:
    st.error(f"❌ Error al conectar o cargar datos: {e}")
