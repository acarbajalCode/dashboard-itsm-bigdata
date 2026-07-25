import streamlit as st
import pymongo
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Dashboard Gerencial ITSM", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

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

    # --- PANEL LATERAL DE FILTROS GLOBALES ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1828/1828859.png", width=50)
    st.sidebar.title("🎛️ Filtros Globales!!")
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
    if mes_sel != "Todos":
        if 'Mes' in df_tendencias.columns: df_tendencias = df_tendencias[df_tendencias['Mes'] == mes_sel]

    if hardware_sel:
        if 'tipo_hardware' in df_brechas.columns: df_brechas = df_brechas[df_brechas['tipo_hardware'].isin(hardware_sel)]
    if garantia_sel:
        if 'estado_garantia' in df_brechas.columns: df_brechas = df_brechas[df_brechas['estado_garantia'].isin(garantia_sel)]

    st.title("📊 Dashboard Ejecutivo ITSM & Analítica Predictiva")
    st.caption("Visión integrada de incidentes, infraestructura y pronóstico impulsado por Machine Learning")
    st.markdown("---")

    # --- PESTAÑAS Y NAVEGACIÓN ---
    pestana = st.radio(
        "Seleccione la vista gerencial:",
        [
            "📈 Visión General y Tendencias", 
            "🛡️ Análisis de Brechas y Hardware", 
            "🔎 Detalle y Drill-Down de Equipos",
            "🔮 Predicción de Demanda (ML)"
        ],
        horizontal=True
    )
    st.markdown("---")

    # --- VISTA 1: TENDENCIAS Y TOP INCIDENTES ---
    if pestana == "📈 Visión General y Tendencias":
        tot_inc_tendencia = int(df_tendencias["Volumen_Mensual"].sum()) if not df_tendencias.empty else 0
        k1, k2 = st.columns(2)
        k1.metric("📌 Total Incidentes (Periodo Seleccionado)", f"{tot_inc_tendencia:,}")
        k2.metric("📋 Categorías Activas", f"{len(df_top) if not df_top.empty else 0}")
        st.markdown("<br>", unsafe_allow_html=True)

        col_izq, col_der = st.columns([6, 4])
        with col_izq:
            st.subheader(f"📈 Evolución Temporal ({anio_sel} - {mes_sel})")
            if not df_tendencias.empty:
                df_curva = df_tendencias.groupby("Anio_Mes")["Volumen_Mensual"].sum().reset_index()
                fig_linea = px.line(df_curva, x="Anio_Mes", y="Volumen_Mensual", markers=True)
                fig_linea.update_traces(line_color="#0066CC")
                st.plotly_chart(fig_linea, use_container_width=True)
        with col_der:
            st.subheader("🔥 Top Incidentes (Histórico)")
            if not df_top.empty:
                df_top_sub = df_top.head(8).sort_values(by="Total_Incidentes", ascending=True)
                fig_barras = px.bar(df_top_sub, x="Total_Incidentes", y="Titulo_Limpio", orientation='h', text="Total_Incidentes")
                st.plotly_chart(fig_barras, use_container_width=True)

    # --- VISTA 2: BRECHAS Y GARANTÍAS ---
    elif pestana == "🛡️ Análisis de Brechas y Hardware":
        st.info("💡 **Nota Gerencial:** El análisis de brechas e infraestructura evalúa el rendimiento histórico total para justificar decisiones de CapEx. No se ve afectado por el filtro de Año/Mes.")
        
        tot_inc_hw = int(df_brechas["Volumen_Incidentes"].sum()) if not df_brechas.empty else 0
        min_per = int(df_brechas["Minutos_Perdidos_Soporte"].sum()) if not df_brechas.empty else 0
        csat_prom = df_brechas["CSAT_Promedio"].mean() if not df_brechas.empty else 0.0

        k1, k2, k3 = st.columns(3)
        k1.metric("📌 Tickets (Infraestructura)", f"{tot_inc_hw:,}")
        k2.metric("⭐ CSAT Promedio Global", f"{csat_prom:.2f} / 5.0")
        k3.metric("⏳ Soporte Perdido Total", f"{min_per:,} min")

        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            if not df_brechas.empty:
                fig_hw = px.bar(df_brechas, x="tipo_hardware", y="Volumen_Incidentes", color="estado_garantia", barmode="group", text_auto=True)
                st.plotly_chart(fig_hw, use_container_width=True)
        with col_g2:
            if not df_brechas.empty:
                fig_bubble = px.scatter(df_brechas, x="tipo_hardware", y="CSAT_Promedio", size="Minutos_Perdidos_Soporte", color="estado_garantia")
                st.plotly_chart(fig_bubble, use_container_width=True)

    # --- VISTA 3: DRILL-DOWN ---
    elif pestana == "🔎 Detalle y Drill-Down de Equipos":
        st.info("💡 **Nota Gerencial:** Los datos presentados aquí reflejan el impacto histórico total del equipo seleccionado.")
        hardware_filtrado_disponible = sorted(df_brechas["tipo_hardware"].unique().tolist()) if not df_brechas.empty else []
        
        if hardware_filtrado_disponible:
            equipo_seleccionado = st.selectbox("👉 Elija un Equipo Específico:", options=hardware_filtrado_disponible)
            if equipo_seleccionado:
                df_detalle_eq = df_brechas[df_brechas["tipo_hardware"] == equipo_seleccionado]
                col_d1, col_d2, col_d3 = st.columns(3)
                col_d1.metric(f"📌 Incidentes", f"{df_detalle_eq['Volumen_Incidentes'].sum():,}")
                col_d2.metric("⭐ CSAT Promedio", f"{df_detalle_eq['CSAT_Promedio'].mean():.2f} / 5.0")
                col_d3.metric("⏳ Minutos Perdidos", f"{df_detalle_eq['Minutos_Perdidos_Soporte'].sum():,} min")
                st.dataframe(df_detalle_eq, use_container_width=True)

    # --- VISTA 4: MACHINE LEARNING (NUEVA) ---
    elif pestana == "🔮 Predicción de Demanda (ML) Nuevo":
        st.subheader("🤖 Algoritmo Predictivo: Facebook Prophet")
        st.markdown("Basado en el historial de tickets, proyectamos la demanda futura para anticipar la carga operativa.")
        
        # Opciones para que tomen decisiones
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            meses_futuros = st.slider("📅 Seleccione los meses a proyectar:", min_value=1, max_value=12, value=3)
        with col_ctrl2:
            st.info(f"El modelo entrenará con toda la base de datos y generará un pronóstico para los próximos **{meses_futuros} meses**.")

        if not df_tendencias_raw.empty:
            # Preparar datos para Prophet (Requiere columnas 'ds' para fecha y 'y' para valor)
            df_ml = df_tendencias_raw.groupby("Anio_Mes")["Volumen_Mensual"].sum().reset_index()
            df_ml['ds'] = pd.to_datetime(df_ml['Anio_Mes'] + '-01') # Convertimos a fecha real
            df_ml = df_ml.rename(columns={'Volumen_Mensual': 'y'})
            
            with st.spinner("🧠 Entrenando modelo predictivo..."):
                # Instanciar y entrenar el modelo
                modelo = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                modelo.fit(df_ml)
                
                # Crear proyecciones
                futuro = modelo.make_future_dataframe(periods=meses_futuros, freq='MS') # MS = Month Start
                prediccion = modelo.predict(futuro)
                
                # Gráfico interactivo
                fig_ml = go.Figure()
                
                # Datos Reales
                fig_ml.add_trace(go.Scatter(x=df_ml['ds'], y=df_ml['y'], mode='lines+markers', name='Demanda Histórica', line=dict(color='blue')))
                
                # Predicción
                fig_ml.add_trace(go.Scatter(x=prediccion['ds'].tail(meses_futuros + 1), y=prediccion['yhat'].tail(meses_futuros + 1), mode='lines+markers', name='Pronóstico (Demanda Esperada)', line=dict(color='orange', dash='dash')))
                
                # Intervalos de confianza (El margen de error)
                fig_ml.add_trace(go.Scatter(
                    x=pd.concat([prediccion['ds'].tail(meses_futuros + 1), prediccion['ds'].tail(meses_futuros + 1)[::-1]]),
                    y=pd.concat([prediccion['yhat_upper'].tail(meses_futuros + 1), prediccion['yhat_lower'].tail(meses_futuros + 1)[::-1]]),
                    fill='toself', fillcolor='rgba(255, 165, 0, 0.2)', line=dict(color='rgba(255,255,255,0)'), name='Margen de Incertidumbre'
                ))
                
                fig_ml.update_layout(title="Proyección de Tickets en el Tiempo", xaxis_title="Fecha", yaxis_title="Volumen de Tickets")
                st.plotly_chart(fig_ml, use_container_width=True)
                
                # Tabla resumen para la toma de decisiones
                st.markdown("#### 📋 Detalle de la Predicción (Siguientes Meses)")
                resumen_pred = prediccion[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(meses_futuros)
                resumen_pred.columns = ['Fecha Estimada', 'Tickets Esperados', 'Escenario Optimista (Mínimo)', 'Escenario Pesimista (Máximo)']
                st.dataframe(resumen_pred, use_container_width=True)

except Exception as e:
    st.error(f"❌ Error al conectar o cargar datos: {e}")
