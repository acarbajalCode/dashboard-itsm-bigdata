import streamlit as st
import pymongo
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

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

    # --- CABECERA PRINCIPAL ---
    st.title("📊 Dashboard Ejecutivo ITSM & Analítica Predictiva")
    st.caption("Visión integrada de incidentes, infraestructura y pronóstico de demanda institucional")
    st.markdown("---")

    # --- PESTAÑAS Y NAVEGACIÓN ---
    pestana = st.radio(
        "Seleccione la vista gerencial:",
        ["📈 Visión General y Tendencias", "🛡️ Análisis de Brechas y Hardware", "🔎 Detalle y Drill-Down de Equipos", "🔮 Predicción de Demanda (ML)"],
        horizontal=True
    )
    st.markdown("---")

    # =========================================================================
    # --- VISTA 1: TENDENCIAS Y TOP INCIDENTES ---
    # =========================================================================
    if pestana == "📈 Visión General y Tendencias":
        tot_inc_tendencia = int(df_tendencias["Volumen_Mensual"].sum()) if not df_tendencias.empty else (int(df_top["Total_Incidentes"].sum()) if not df_top.empty else 0)
        k1, k2 = st.columns(2)
        k1.metric("📌 Total Incidentes (Periodo Seleccionado)", f"{tot_inc_tendencia:,}")
        k2.metric("📋 Categorías Activas Registradas", f"{len(df_top) if not df_top.empty else 0}")
        st.markdown("<br>", unsafe_allow_html=True)

        col_izq, col_der = st.columns([6, 4])
        with col_izq:
            st.subheader(f"📈 Evolución Temporal ({anio_sel} - {mes_sel})")
            if not df_tendencias.empty:
                df_curva = df_tendencias.groupby("Anio_Mes")["Volumen_Mensual"].sum().reset_index()
                fig_linea = px.line(df_curva, x="Anio_Mes", y="Volumen_Mensual", markers=True)
                fig_linea.update_traces(line_color="#0066CC")
                fig_linea.update_layout(plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Año-Mes", yaxis_title="Volumen de Tickets")
                st.plotly_chart(fig_linea, use_container_width=True)
                
        with col_der:
            st.subheader("🔥 Ranking de Incidentes Recurrentes")
            if not df_top.empty:
                max_top = len(df_top)
                top_n = st.slider("Ajustar cantidad a visualizar en el gráfico:", min_value=3, max_value=max_top, value=10)
                df_top_sub = df_top.head(top_n).sort_values(by="Total_Incidentes", ascending=True)
                altura_dinamica = max(400, top_n * 35) 
                fig_barras = px.bar(
                    df_top_sub, x="Total_Incidentes", y="Titulo_Limpio", orientation='h', 
                    text="Total_Incidentes", color="Total_Incidentes", color_continuous_scale="Blues"
                )
                fig_barras.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=altura_dinamica, showlegend=False, yaxis_title="")
                st.plotly_chart(fig_barras, use_container_width=True)
                
                with st.expander("📋 Ver el listado histórico completo"):
                    st.dataframe(df_top, use_container_width=True)

    # =========================================================================
    # --- VISTA 2: BRECHAS Y GARANTÍAS ---
    # =========================================================================
    elif pestana == "🛡️ Análisis de Brechas y Hardware":
        st.info("💡 **Nota Gerencial:** El análisis de infraestructura evalúa el rendimiento histórico total para justificar decisiones. No se ve afectado por el filtro de Año/Mes.")
        tot_inc_hw = int(df_brechas["Volumen_Incidentes"].sum()) if not df_brechas.empty else 0
        min_per = int(df_brechas["Minutos_Perdidos_Soporte"].sum()) if not df_brechas.empty else 0
        csat_prom = df_brechas["CSAT_Promedio"].mean() if not df_brechas.empty else 0.0

        if not df_brechas.empty and "estado_garantia" in df_brechas.columns:
            df_sin_garantia = df_brechas[df_brechas["estado_garantia"].astype(str).str.contains("Vencid|Sin|Expirad|No", case=False, na=False)]
            tickets_sin_garantia = int(df_sin_garantia["Volumen_Incidentes"].sum())
        else:
            tickets_sin_garantia = 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📌 Reportes Totales", f"{tot_inc_hw:,}")
        k2.metric("⭐ Nivel de Satisfacción", f"{csat_prom:.2f} / 5.0")
        k3.metric("⏱️ Minutos Perdidos", f"{min_per:,} min")
        k4.metric("⚠️ Fallas Sin Garantía", f"{tickets_sin_garantia:,}", delta="Costo asumido por la Institución", delta_color="inverse")
        st.markdown("<br>", unsafe_allow_html=True)

        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            if not df_brechas.empty:
                st.markdown("#### 📊 Volumen de Fallas por Equipo y Cobertura")
                fig_hw = px.bar(
                    df_brechas, x="tipo_hardware", y="Volumen_Incidentes", 
                    color="estado_garantia", barmode="group", text_auto=True,
                    color_discrete_sequence=["#0066CC", "#FF4B4B", "#00CC66"]
                )
                fig_hw.update_layout(plot_bgcolor="rgba(0,0,0,0)", legend_title="Estado de Garantía", margin=dict(t=20), xaxis_title="", yaxis_title="N° Fallas")
                st.plotly_chart(fig_hw, use_container_width=True)
                
        with col_g2:
            if not df_brechas.empty:
                st.markdown("#### ⚖️ Impacto del Desgaste (Satisfacción vs Tiempo Perdido)")
                fig_bubble = px.scatter(
                    df_brechas, x="tipo_hardware", y="CSAT_Promedio", 
                    size="Minutos_Perdidos_Soporte", color="estado_garantia",
                    color_discrete_sequence=["#0066CC", "#FF4B4B", "#00CC66"]
                )
                fig_bubble.update_layout(plot_bgcolor="rgba(0,0,0,0)", legend_title="Estado de Garantía", margin=dict(t=20), xaxis_title="", yaxis_title="Satisfacción (CSAT)")
                st.plotly_chart(fig_bubble, use_container_width=True)

    # =========================================================================
    # --- VISTA 3: DRILL-DOWN Y COBERTURA DE GARANTÍA ---
    # =========================================================================
    elif pestana == "🔎 Detalle y Drill-Down de Equipos":
        st.info("💡 **Nota Gerencial:** Seleccione un equipo específico para conocer si sus incidentes recientes están cubiertos por un proveedor o si representan carga exclusiva para nuestros técnicos.")
        hardware_filtrado_disponible = sorted(df_brechas["tipo_hardware"].unique().tolist()) if not df_brechas.empty else []
        
        if hardware_filtrado_disponible:
            equipo_seleccionado = st.selectbox("👉 Elija un Equipo Específico:", options=hardware_filtrado_disponible)
            
            if equipo_seleccionado:
                df_detalle_eq = df_brechas[df_brechas["tipo_hardware"] == equipo_seleccionado]
                
                col_d1, col_d2, col_d3 = st.columns(3)
                col_d1.metric(f"📌 Incidentes en {equipo_seleccionado}", f"{df_detalle_eq['Volumen_Incidentes'].sum():,}")
                col_d2.metric("⭐ Satisfacción Promedio", f"{df_detalle_eq['CSAT_Promedio'].mean():.2f} / 5.0")
                col_d3.metric("⏱️ Minutos Perdidos", f"{df_detalle_eq['Minutos_Perdidos_Soporte'].sum():,} min")
                
                st.markdown("---")
                col_graf, col_tabla = st.columns([4, 6])
                
                with col_graf:
                    st.markdown(f"#### 🛡️ Estado de Garantía ({equipo_seleccionado})")
                    df_garantia = df_detalle_eq.groupby("estado_garantia")["Volumen_Incidentes"].sum().reset_index()
                    fig_pie = px.pie(
                        df_garantia, values="Volumen_Incidentes", names="estado_garantia", 
                        hole=0.45, color="estado_garantia",
                        color_discrete_sequence=["#0066CC", "#FF4B4B", "#00CC66"]
                    )
                    fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True, legend=dict(orientation="h", y=-0.2))
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with col_tabla:
                    st.markdown("#### 📋 Desglose Operativo y Proveedores")
                    columnas_mostrar = ["estado_garantia", "Volumen_Incidentes", "CSAT_Promedio", "Minutos_Perdidos_Soporte"]
                    df_mostrar = df_detalle_eq[columnas_mostrar].rename(columns={
                        "estado_garantia": "Condición de Garantía",
                        "Volumen_Incidentes": "N° de Fallas",
                        "CSAT_Promedio": "Nota (CSAT)",
                        "Minutos_Perdidos_Soporte": "Min. Perdidos"
                    })
                    st.dataframe(df_mostrar, use_container_width=True, hide_index=True)

    # =========================================================================
    # --- VISTA 4: MACHINE LEARNING (FÁCIL Y GERENCIAL) ---
    # =========================================================================
    elif pestana == "🔮 Predicción de Demanda (ML)":
        st.subheader("🤖 Pronóstico y Recomendaciones Gerenciales")
        st.markdown("Proyección de carga operativa para la institución, diseñada para anticipar problemas e interrupciones.")
        
        col_persp, col_ctrl1, col_ctrl2 = st.columns([1.5, 1, 2])
        with col_persp:
            perspectiva = st.radio("🔍 ¿Qué desea analizar?", ["🖥️ Equipos Físicos (Hardware)", "📂 Tipos de Problema (Incidentes)"])
        with col_ctrl1:
            meses_futuros = st.slider("📅 Meses a proyectar:", min_value=1, max_value=12, value=3)
        with col_ctrl2:
            if perspectiva == "🖥️ Equipos Físicos (Hardware)":
                opciones_pred = ["Visión Global"] + sorted(df_brechas_raw["tipo_hardware"].unique().tolist()) if not df_brechas_raw.empty else ["Visión Global"]
            else:
                opciones_pred = ["Visión Global"] + sorted(df_top_raw["Titulo_Limpio"].unique().tolist()) if not df_top_raw.empty else ["Visión Global"]
            
            categoria_pred = st.selectbox(f"🎯 Seleccione la categoría específica:", options=opciones_pred)

        if not df_tendencias_raw.empty:
            df_ml = df_tendencias_raw.groupby("Anio_Mes")["Volumen_Mensual"].sum().reset_index()
            df_ml['ds'] = pd.to_datetime(df_ml['Anio_Mes'] + '-01')
            df_ml = df_ml.rename(columns={'Volumen_Mensual': 'y'})
            
            with st.spinner("🧠 Generando proyecciones..."):
                modelo = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                modelo.fit(df_ml)
                futuro = modelo.make_future_dataframe(periods=meses_futuros, freq='MS')
                prediccion = modelo.predict(futuro)
                
                fig_ml = go.Figure()
                fig_ml.add_trace(go.Scatter(x=df_ml['ds'], y=df_ml['y'], mode='lines+markers', name='Demanda Histórica', line=dict(color='#0066CC', width=3)))
                fig_ml.add_trace(go.Scatter(x=prediccion['ds'].tail(meses_futuros + 1), y=prediccion['yhat'].tail(meses_futuros + 1), mode='lines+markers', name='Carga Esperada', line=dict(color='#FF9900', dash='dash', width=3)))
                fig_ml.add_trace(go.Scatter(
                    x=pd.concat([prediccion['ds'].tail(meses_futuros + 1), prediccion['ds'].tail(meses_futuros + 1)[::-1]]),
                    y=pd.concat([prediccion['yhat_upper'].tail(meses_futuros + 1), prediccion['yhat_lower'].tail(meses_futuros + 1)[::-1]]),
                    fill='toself', fillcolor='rgba(255, 153, 0, 0.15)', line=dict(color='rgba(255,255,255,0)'), name='Margen de variación'
                ))
                fig_ml.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_ml, use_container_width=True)

                tot_tickets_proyectados = int(prediccion['yhat'].tail(meses_futuros).sum())
                
                # PERSPECTIVA: HARDWARE
                if perspectiva == "🖥️ Equipos Físicos (Hardware)":
                    df_brechas_clean = df_brechas_raw.copy()
                    df_brechas_clean["Volumen_Incidentes"] = pd.to_numeric(df_brechas_clean["Volumen_Incidentes"], errors='coerce').fillna(0)
                    df_brechas_clean["Minutos_Perdidos_Soporte"] = pd.to_numeric(df_brechas_clean["Minutos_Perdidos_Soporte"], errors='coerce').fillna(0)
                    df_brechas_clean["CSAT_Promedio"] = pd.to_numeric(df_brechas_clean["CSAT_Promedio"], errors='coerce').fillna(0)

                    if categoria_pred == "Visión Global":
                        cuota = 1.0
                        total_vol = df_brechas_clean["Volumen_Incidentes"].sum()
                        factor_tiempo = df_brechas_clean["Minutos_Perdidos_Soporte"].sum() / total_vol if total_vol > 0 else 0
                        csat_promedio = df_brechas_clean["CSAT_Promedio"].mean()
                    else:
                        df_cat = df_brechas_clean[df_brechas_clean["tipo_hardware"] == categoria_pred]
                        tot_cat = df_cat["Volumen_Incidentes"].sum()
                        total_vol_global = df_brechas_clean["Volumen_Incidentes"].sum()
                        cuota = tot_cat / total_vol_global if total_vol_global > 0 else 0
                        factor_tiempo = df_cat["Minutos_Perdidos_Soporte"].sum() / tot_cat if tot_cat > 0 else 0
                        csat_promedio = df_cat["CSAT_Promedio"].mean()

                    tickets_esperados_cat = int(tot_tickets_proyectados * cuota)
                    minutos_totales = round((tickets_esperados_cat * factor_tiempo), 1)

                    st.markdown(f"### 📊 Detalle de Proyección: **{categoria_pred}**")
                    k1, k2, k3 = st.columns(3)
                    k1.metric("📌 Reportes Estimados", f"{tickets_esperados_cat:,} casos")
                    k2.metric("⭐ Nivel de Satisfacción", f"{csat_promedio:.2f} / 5.0")
                    k3.metric("⏱️ Tiempo de Soporte Estimado", f"{minutos_totales} min")

                    alerta_csat = "crítico" if csat_promedio < 3.0 else "estable"
                    
                    st.markdown("---")
                    st.markdown("### 📋 Recomendaciones para la Toma de Decisiones")
                    
                    if categoria_pred != "Visión Global":
                        if alerta_csat == "crítico":
                            st.warning(f"⚠️ **Riesgo de Productividad:** Los problemas con **{categoria_pred}** están generando frustración en los usuarios y mala calificación del servicio respuesta. \n\n**Sugerencia:** Evaluar el reemplazo de estos equipos o exigir garantías. El tiempo y esfuerzo que el personal invierte en intentar repararlos constantemente afecta la continuidad del trabajo institucional.")
                        else:
                            st.success(f"✅ **Estado Controlado:** La operatividad de los usuarios con **{categoria_pred}** se mantiene estable y predecible. \n\n**Sugerencia:** Continuar con los mantenimientos periódicos normales para asegurar que la buena experiencia de los usuarios se mantenga en el tiempo.")
                    else:
                        st.info("💡 **Visión General de Equipos:** Se sugiere enfocar los mantenimientos preventivos en los equipos que consumen la mayor cantidad de 'Tiempo de Soporte Estimado', ya que son los que causan más interrupciones al trabajo diario de la institución.")

                # PERSPECTIVA: TIPO DE INCIDENTE
                else:
                    df_top_clean = df_top_raw.copy()
                    df_top_clean["Total_Incidentes"] = pd.to_numeric(df_top_clean["Total_Incidentes"], errors='coerce').fillna(0)
                    
                    if categoria_pred == "Visión Global":
                        cuota = 1.0
                    else:
                        df_cat = df_top_clean[df_top_clean["Titulo_Limpio"] == categoria_pred]
                        tot_cat = df_cat["Total_Incidentes"].sum()
                        total_vol_global = df_top_clean["Total_Incidentes"].sum()
                        cuota = tot_cat / total_vol_global if total_vol_global > 0 else 0

                    tickets_esperados_cat = int(tot_tickets_proyectados * cuota)
                    porcentaje_impacto = round((cuota * 100), 1)

                    st.markdown(f"### 📊 Detalle de Proyección: **{categoria_pred}**")
                    k1, k2 = st.columns(2)
                    k1.metric("📌 Casos a Atender", f"{tickets_esperados_cat:,} casos")
                    k2.metric("📉 Volumen sobre el Total", f"{porcentaje_impacto}%")

                    st.markdown("---")
                    st.markdown("### 📋 Recomendaciones para la Toma de Decisiones")
                    
                    if categoria_pred != "Visión Global":
                        if porcentaje_impacto > 15.0:
                            st.warning(f"⚠️ **Problema Frecuente:** El incidente '**{categoria_pred}**' consumirá una gran parte del tiempo del personal de soporte técnico. \n\n**Sugerencia:** Para no retrasar el trabajo en la Empresa, se recomienda crear guías o manuales rápidos para que los usuarios aprendan a solucionar esto por su cuenta sin necesidad de esperar a un técnico.")
                        elif porcentaje_impacto >= 5.0:
                            st.info(f"💡 **Oportunidad de Mejora:** El reporte de '**{categoria_pred}**' ocurre con cierta regularidad. \n\n**Sugerencia:** Asegurarse de que todo el equipo de TI conozca la forma más rápida de solucionarlo para evitar que los usuarios permanezcan inactivos por mucho tiempo.")
                        else:
                            st.success(f"✅ **Impacto Menor:** El problema '**{categoria_pred}**' no es frecuente y su impacto en la institución es mínimo. Continuar con los flujos de atención habituales.")
                    else:
                         st.info("💡 **Visión General de Problemas:** Se recomienda identificar cuáles son los problemas más repetitivos e implementar manuales simples para los usuarios institucionales. Esto agilizará enormemente el trabajo de todos.")

except Exception as e:
    st.error(f"❌ Error al conectar o procesar datos predictivos: {e}")
