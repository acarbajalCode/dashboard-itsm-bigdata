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
            st.subheader("🔥 Ranking de Incidentes")
            if not df_top.empty:
                # 1. Slider interactivo para elegir cuántos incidentes graficar
                max_top = len(df_top)
                top_n = st.slider("Ajustar cantidad a visualizar en el gráfico:", min_value=3, max_value=max_top, value=10)
                
                df_top_sub = df_top.head(top_n).sort_values(by="Total_Incidentes", ascending=True)
                
                # --- EL TRUCO MAGICO: Altura Dinámica ---
                # Le damos 35 pixeles de altura garantizada a cada barra. Si son pocas, el mínimo es 400px.
                altura_dinamica = max(400, top_n * 35) 

                fig_barras = px.bar(
                    df_top_sub, x="Total_Incidentes", y="Titulo_Limpio", orientation='h', 
                    text="Total_Incidentes", color="Total_Incidentes", color_continuous_scale="Blues"
                )
                
                # Aplicamos la altura dinámica al layout
                fig_barras.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=altura_dinamica, showlegend=False)
                st.plotly_chart(fig_barras, use_container_width=True)
                
                # 2. Expansor para ver el 100% de la data cruda
                with st.expander("📋 Ver el listado histórico completo (Todas las categorías)"):
                    st.dataframe(df_top, use_container_width=True)

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

   # --- VISTA 4: MACHINE LEARNING (PREDICCIÓN Y TOMA DE DECISIONES) ---
    elif pestana == "🔮 Predicción de Demanda (ML)":
        st.subheader("🤖 Pronóstico y Plan de Acción Estratégico")
        st.markdown("Proyección de demanda categorizada y cálculo de Horas-Hombre (HH) en riesgo para justificación de CapEx/OpEx.")
        
        # 1. CONTROLES DE DECISIÓN
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            meses_futuros = st.slider("📅 Meses a proyectar en el futuro:", min_value=1, max_value=12, value=3)
        with col_ctrl2:
            opciones_pred = ["Impacto Global (Toda la Infraestructura)"] + sorted(df_brechas_raw["tipo_hardware"].unique().tolist()) if not df_brechas_raw.empty and "tipo_hardware" in df_brechas_raw.columns else ["Impacto Global (Toda la Infraestructura)"]
            categoria_pred = st.selectbox("🎯 Categorizar predicción para toma de decisiones en:", options=opciones_pred)

        if not df_tendencias_raw.empty:
            # 2. PREPARACIÓN Y MOTOR PROPHET
            df_ml = df_tendencias_raw.groupby("Anio_Mes")["Volumen_Mensual"].sum().reset_index()
            df_ml['ds'] = pd.to_datetime(df_ml['Anio_Mes'] + '-01')
            df_ml = df_ml.rename(columns={'Volumen_Mensual': 'y'})
            
            with st.spinner("🧠 Calculando proyecciones e impacto operativo..."):
                modelo = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                modelo.fit(df_ml)
                futuro = modelo.make_future_dataframe(periods=meses_futuros, freq='MS')
                prediccion = modelo.predict(futuro)
                
                # 3. GRÁFICO VISUAL
                fig_ml = go.Figure()
                fig_ml.add_trace(go.Scatter(x=df_ml['ds'], y=df_ml['y'], mode='lines+markers', name='Histórico Real', line=dict(color='#0066CC', width=3)))
                fig_ml.add_trace(go.Scatter(x=prediccion['ds'].tail(meses_futuros + 1), y=prediccion['yhat'].tail(meses_futuros + 1), mode='lines+markers', name='Tendencia Esperada', line=dict(color='#FF9900', dash='dash', width=3)))
                fig_ml.add_trace(go.Scatter(
                    x=pd.concat([prediccion['ds'].tail(meses_futuros + 1), prediccion['ds'].tail(meses_futuros + 1)[::-1]]),
                    y=pd.concat([prediccion['yhat_upper'].tail(meses_futuros + 1), prediccion['yhat_lower'].tail(meses_futuros + 1)[::-1]]),
                    fill='toself', fillcolor='rgba(255, 153, 0, 0.15)', line=dict(color='rgba(255,255,255,0)'), name='Margen de Incertidumbre'
                ))
                fig_ml.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_ml, use_container_width=True)

                # 4. LIMPIEZA DE DATOS Y MATEMÁTICA (Solución del error '0')
                # Forzamos que las columnas sean numéricas para evitar errores de cálculo
                df_brechas_clean = df_brechas_raw.copy()
                df_brechas_clean["Volumen_Incidentes"] = pd.to_numeric(df_brechas_clean["Volumen_Incidentes"], errors='coerce').fillna(0)
                df_brechas_clean["Minutos_Perdidos_Soporte"] = pd.to_numeric(df_brechas_clean["Minutos_Perdidos_Soporte"], errors='coerce').fillna(0)

                tot_tickets_proyectados = int(prediccion['yhat'].tail(meses_futuros).sum())
                
                # Cálculo de cuotas y promedios seguros
                if categoria_pred == "Impacto Global (Toda la Infraestructura)":
                    cuota = 1.0
                    total_vol = df_brechas_clean["Volumen_Incidentes"].sum()
                    min_promedio = df_brechas_clean["Minutos_Perdidos_Soporte"].sum() / total_vol if total_vol > 0 else 0
                else:
                    df_cat = df_brechas_clean[df_brechas_clean["tipo_hardware"] == categoria_pred]
                    tot_cat = df_cat["Volumen_Incidentes"].sum()
                    total_vol_global = df_brechas_clean["Volumen_Incidentes"].sum()
                    
                    cuota = tot_cat / total_vol_global if total_vol_global > 0 else 0
                    min_promedio = df_cat["Minutos_Perdidos_Soporte"].sum() / tot_cat if tot_cat > 0 else 0

                # Formateo final
                tickets_esperados_cat = int(tot_tickets_proyectados * cuota)
                horas_hombre_perdidas = int((tickets_esperados_cat * min_promedio) / 60)
                min_promedio_display = round(min_promedio, 1)

                # 5. MÉTRICAS DE IMPACTO
                st.markdown(f"### 📊 Impacto Operativo para: **{categoria_pred}** ({meses_futuros} meses)")
                k1, k2, k3 = st.columns(3)
                k1.metric(label="📌 Tickets Proyectados", value=f"{tickets_esperados_cat:,}")
                k2.metric(label="⏱️ Tiempo Prom. por Ticket", value=f"{min_promedio_display} min")
                k3.metric(label="🔥 Horas-Hombre en Riesgo", value=f"{horas_hombre_perdidas:,} HH", delta="Tiempo improductivo", delta_color="inverse")

                # 6. RECOMENDACIONES CORPORATIVAS AUTOMATIZADAS
                st.markdown("---")
                st.markdown("### 📋 Recomendaciones para Jefatura y Gerencia de TI")
                
                if categoria_pred == "Impacto Global (Toda la Infraestructura)":
                    st.info(f"**Análisis Estratégico Global:** Se proyecta una pérdida total de **{horas_hombre_perdidas} Horas-Hombre** en los próximos {meses_futuros} meses debido a soporte de hardware. Se recomienda a la jefatura evaluar el presupuesto (CapEx) para renovar los equipos con mayor incidencia y reestructurar la asignación de tickets en el equipo técnico (OpEx) para evitar saturación y cuellos de botella en la atención diaria.")
                else:
                    st.warning(f"**Alerta Operativa sobre {categoria_pred}:** Mantener la tendencia actual en este componente específico representará una fuga de **{horas_hombre_perdidas} Horas-Hombre**. El tiempo promedio de soporte ({min_promedio_display} minutos por ticket) impacta negativamente en los SLA del Service Desk. \n\n**Plan de Acción Sugerido:** \n1. **CapEx:** Renovar prioritariamente el inventario de {categoria_pred} que cuente con garantías vencidas.\n2. **OpEx:** Programar un ciclo de mantenimiento preventivo masivo enfocado exclusivamente en estos equipos.\n3. **Procesos:** Escalar de manera automatizada los incidentes repetitivos de esta categoría a nivel 2 de soporte para liberar la primera línea.")

except Exception as e:
    st.error(f"❌ Error al conectar o procesar datos predictivos: {e}")
