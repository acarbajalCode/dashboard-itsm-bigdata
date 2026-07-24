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
    # Reemplaza st.secrets["MONGO_URI"] por tu URI entre comillas si pruebas localmente
    cliente = pymongo.MongoClient(st.secrets["MONGO_URI"])
    db = cliente["itsm_analytics"]
    
    df_top = pd.DataFrame(list(db["top_incidentes"].find())).drop(columns=["_id"], errors='ignore')
    df_brechas = pd.DataFrame(list(db["brechas_servicio"].find())).drop(columns=["_id"], errors='ignore')
    df_tendencias = pd.DataFrame(list(db["tendencias_temporales"].find())).drop(columns=["_id"], errors='ignore')
    
    return df_top, df_brechas, df_tendencias

try:
    df_top_raw, df_brechas_raw, df_tendencias_raw = cargar_datos_mongo()

    # --- PANEL LATERAL DE FILTROS (TIPO POWER BI SLICERS) ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1828/1828859.png", width=50)
    st.sidebar.title("🎛️ Filtros Interactivos")
    st.sidebar.markdown("*Ajuste los combos para filtrar todo el dashboard:*")
    st.sidebar.markdown("---")

    # 1. Filtro de Periodo / Año-Mes
    periodos_disponibles = sorted(df_tendencias_raw["Anio_Mes"].unique().tolist()) if not df_tendencias_raw.empty else []
    periodos_sel = st.sidebar.multiselect(
        "📅 Seleccionar Año - Mes:",
        options=periodos_disponibles,
        default=periodos_disponibles # Por defecto selecciona todos
    )

    # 2. Filtro por Tipo de Hardware
    hardware_disponible = sorted(df_brechas_raw["tipo_hardware"].unique().tolist()) if not df_brechas_raw.empty else []
    hardware_sel = st.sidebar.multiselect(
        "💻 Tipo de Hardware:",
        options=hardware_disponible,
        default=hardware_disponible
    )

    # 3. Filtro por Garantía
    garantia_disponible = sorted(df_brechas_raw["estado_garantia"].unique().tolist()) if not df_brechas_raw.empty else []
    garantia_sel = st.sidebar.multiselect(
        "🛡️ Estado de Garantía:",
        options=garantia_disponible,
        default=garantia_disponible
    )

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip Power BI:** Los filtros aplicados actualizan métricas, gráficos y tablas automáticamente.")

    # --- APLICACIÓN DE FILTROS A LOS DATAFRAMES ---
    df_tendencias = df_tendencias_raw[df_tendencias_raw["Anio_Mes"].isin(periodos_sel)] if not df_tendencias_raw.empty else pd.DataFrame()
    df_brechas = df_brechas_raw[
        (df_brechas_raw["tipo_hardware"].isin(hardware_sel)) & 
        (df_brechas_raw["estado_garantia"].isin(garantia_sel))
    ] if not df_brechas_raw.empty else pd.DataFrame()
    df_top = df_top_raw if not df_top_raw.empty else pd.DataFrame()

    # --- ENCABEZADO Y KPIS DINÁMICOS ---
    st.title("📊 Dashboard Ejecutivo ITSM & Analítica Predictiva")
    st.caption("Visión integrada de incidentes, infraestructura y satisfacción en tiempo real")

    # Métricas dinámicas calculadas según los filtros
    tot_inc = int(df_tendencias["Volumen_Mensual"].sum()) if not df_tendencias.empty else int(df_top["Total_Incidentes"].sum())
    min_per = int(df_brechas["Minutos_Perdidos_Soporte"].sum()) if not df_brechas.empty else 0
    csat_prom = df_brechas["CSAT_Promedio"].mean() if not df_brechas.empty else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📌 Total Incidentes Filtrados", f"{tot_inc:,}")
    k2.metric("⭐ CSAT Promedio", f"{csat_prom:.2f} / 5.0")
    k3.metric("⏳ Soporte Perdido (Min)", f"{min_per:,} min")
    k4.metric("🖥️ Tipos Hardware Evaluados", f"{len(hardware_sel)} / {len(hardware_disponible)}")

    st.markdown("---")

    # --- PESTAÑAS PRINCIPALES ---
    tab1, tab2, tab3 = st.columns([1, 1, 1])

    # NAVEGACIÓN MEDIANTE TABS INTERACTIVOS
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
            st.subheader("📈 Evolución Temporal de Incidentes")
            if not df_tendencias.empty:
                df_curva = df_tendencias.groupby("Anio_Mes")["Volumen_Mensual"].sum().reset_index()
                fig_linea = px.line(
                    df_curva, x="Anio_Mes", y="Volumen_Mensual",
                    markers=True, text="Volumen_Mensual",
                    title="Tendencia Mensual de Tickets (Filtro Aplicado)",
                    labels={"Anio_Mes": "Periodo", "Volumen_Mensual": "Tickets"}
                )
                fig_linea.update_traces(textposition="top center", line_color="#0066CC")
                fig_linea.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=380)
                st.plotly_chart(fig_linea, use_container_width=True)
            else:
                st.warning("Sin datos para los periodos seleccionados.")

        with col_der:
            st.subheader("🔥 Top Cuellos de Botella")
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

        with col_g2:
            st.subheader("⭐ Satisfacción del Cliente (CSAT) vs Tiempo Perdido")
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
        st.subheader("🔍 Explorador de Detalle (Drill-Down Estilo Power BI)")
        st.markdown("Seleccione un hardware específico para abrir la ficha técnica desglosada:")

        # Combo selector de drill-down
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

            col_d1.metric("Incidentes Totales", f"{tot_inc_eq:,}")
            col_d2.metric("Minutos de Soporte Consumidos", f"{min_perd_eq:,} min")
            col_d3.metric("Calificación CSAT", f"{csat_eq:.2f} / 5.0")

            st.markdown("#### Tabla Desglosada por Estado de Garantía:")
            st.dataframe(
                df_detalle_eq.style.highlight_max(axis=0, color="#ffcccc"),
                use_container_width=True
            )

except Exception as e:
    st.error(f"❌ Error al conectar o cargar datos: {e}")
